"""因果 ODE init 的过拟合诊断:训练集 loss vs held-out loss。

背景:训练是 glob 整个 output/ode_pairs,没有留验证集;而 eff bsz=8、可训 453M、
轨迹池从 37 条长到 ~1700(早期反复啃前几十条),约 23 个 epoch —— 有真实的记忆风险。
本脚本用**完全没进过训练**的一批轨迹(独立 seed 生成到 output/ode_pairs_val)对比。

判据:
  val/train 比值 ≈ 1.0  → 没过拟合,数据量够
  比值 1.1~1.3         → 轻度,可接受(继续训要盯着)
  比值 > 1.5           → 明显记忆,该加数据或早停

为降低随机性,σ 档位组合固定 seed,train/val 用**同一组** σ 组合。
用法: CUDA_VISIBLE_DEVICES=1 python scripts/val/eval_ode_overfit.py \
        --ckpt output/ode_init_causal/odeinit_step_2500.pt
"""
import sys, os, glob, argparse
import torch
import torch.nn.functional as F
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--train_dir", default="output/ode_pairs")
ap.add_argument("--val_dir", default="output/ode_pairs_val")
ap.add_argument("--n", type=int, default=48, help="每边取多少条")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
k = torch.load(a.ckpt, map_location="cpu")
denu.load_state_dict(k["denoising_unet"], strict=False); denu.eval()
ctrl = TemporalCausalControl(denu, block_size=a.block, window=0)
ctrl.set_rope(True); ctrl.set_mode("train")          # 与训练同构:并行 block-causal
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
print(f"[ckpt] {a.ckpt} step={k.get('step')}", flush=True)


@torch.no_grad()
def loss_on(files, tag):
    tot, n = 0.0, 0
    g = torch.Generator(device="cpu"); g.manual_seed(a.seed)   # train/val 用同一串 σ 组合
    for f in files:
        d = torch.load(f, map_location="cpu")
        traj = d["traj"].detach().unsqueeze(0).to(dev, dt)      # [1,5,4,F,H,W]
        B, _, C, F_, H, W = traj.shape
        target = traj[:, -1]
        sig_list = torch.tensor(d["student_sigmas"], device=dev)
        motion = d["motion"].detach().to(dev, dt)
        nblk = (F_ + a.block - 1) // a.block
        idx_blk = torch.randint(0, len(sig_list), (B, nblk), generator=g).to(dev)
        idx = idx_blk.repeat_interleave(a.block, dim=1)[:, :F_]
        z = torch.gather(traj, 1, idx.view(B, 1, 1, F_, 1, 1)
                         .expand(B, 1, C, F_, H, W)).squeeze(1)
        sig = sig_list[idx]
        rw.clear()
        refu(d["ref_latent"].detach().to(dev, dt), torch.zeros((), device=dev).long(),
             encoder_hidden_states=d["clip_emb"].detach().to(dev, dt), return_dict=False)
        rr.update(rw, dtype=dt)
        v = denu(z, (sig * 1000.0).to(dt),
                 encoder_hidden_states=[d["clip_emb"].detach().to(dev, dt), motion],
                 pose_cond_fea=None, return_dict=False)[0]
        x0 = z.float() - sig.view(B, 1, F_, 1, 1) * v.float()
        tot += F.mse_loss(x0, target.float()).item(); n += 1
        if n % 16 == 0:
            print(f"  [{tag}] {n}/{len(files)} running={tot/n:.5f}", flush=True)
    return tot / max(n, 1)


tr = sorted(glob.glob(os.path.join(a.train_dir, "*.pt")))[:a.n]
va = sorted(glob.glob(os.path.join(a.val_dir, "*.pt")))[:a.n]
assert va, f"{a.val_dir} 为空,先生成 held-out 轨迹"
print(f"[data] train={len(tr)} 条(已训过)  val={len(va)} 条(全新)", flush=True)
l_tr = loss_on(tr, "train")
l_va = loss_on(va, "val")
r = l_va / l_tr
print(f"\n{'train loss':>14} = {l_tr:.5f}")
print(f"{'val loss':>14} = {l_va:.5f}")
print(f"{'val/train':>14} = {r:.3f}")
verdict = ("✅ 未过拟合" if r < 1.1 else
           "⚠️ 轻度过拟合,可接受但需盯" if r < 1.5 else
           "❌ 明显记忆,需加数据/早停")
print(f"{'判定':>14} : {verdict}")
print("DONE", flush=True)

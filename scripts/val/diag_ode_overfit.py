"""区分「ODE init 过拟合」与「成功继承 teacher 偏差」。

现象:训练 loss 降(0.090→0.064)而验证 x0mse 升(0.1525@200→0.1622@1000)。
但两者测的不是一回事 ——
  训练 loss:  输入 teacher 轨迹点, 目标 teacher 终点
  验证 x0mse: 输入 GT 加新鲜噪声,  目标 GT
所以"训练降验证升"有两种解释,处方相反:
  ① 过拟合           → 多生成轨迹有用
  ② 在逼近不完美 teacher → 多生成轨迹无用,天花板在 teacher

判据:在**同分布的 held-out 轨迹**上算**训练目标本身**的 loss。
  随训练上升 → 过拟合;下降 → 继承 teacher 偏差。
同时在训练集轨迹上算同一指标,两者之差 = 泛化 gap。

用法:
  python scripts/val/diag_ode_overfit.py --gpu 5 \
    --train_pairs output/ode_pairs_sh3 --heldout_pairs output/ode_pairs_sh3_heldout \
    --ckpts output/flow_stage2_sh3/stage2_step_2500.pt \
            output/ode_init_sh3/odeinit_step_{500,1000}.pt
"""
import os, sys, glob, argparse
import numpy as np, torch
sys.path += ["/media/ps/ssd5/ayr/x-nemo-inference", "/media/ps/ssd5/ayr/motar"]
from omegaconf import OmegaConf
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl
XN = "/media/ps/ssd5/ayr/x-nemo-inference"

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--train_pairs", default=f"{XN}/output/ode_pairs_sh3")
ap.add_argument("--heldout_pairs", default=f"{XN}/output/ode_pairs_sh3_heldout")
ap.add_argument("--n", type=int, default=80, help="每边取多少条")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--ckpts", nargs="+", required=True)
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); dt = torch.bfloat16; torch.cuda.set_device(dev)

cfg = OmegaConf.load(f"{XN}/configs/test_ar_model.yaml"); ic = OmegaConf.load(cfg.inference_config)
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet","reference_unet"),
                                map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
ctrl = TemporalCausalControl(denu, block_size=a.block, window=0); ctrl.set_rope(True); ctrl.set_mode("train")
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read",  batch_size=1, fusion_blocks="full")

def load(d, n):
    fs = sorted(glob.glob(f"{d}/*.pt"))[:n]
    return [torch.load(f, map_location="cpu") for f in fs]
sets = {"训练集": load(a.train_pairs, a.n), "held-out": load(a.heldout_pairs, a.n)}
for k, v in sets.items(): print(f"[data] {k}: {len(v)} 条", flush=True)

@torch.no_grad()
def loss_on(items, seed=0):
    """与 ode_init_causal.py 训练目标逐行一致:每 block 独立抽档 → 预测 x0 → 与 teacher 终点比。
    ★ σ 档分配用固定种子,保证跨 ckpt 可比。"""
    g = torch.Generator().manual_seed(seed); acc = []
    for d in items:
        traj = d["traj"].to(dev, dt).unsqueeze(0)                 # [1,5,4,F,H,W]
        B, _, C, F_, H, W = traj.shape
        target = traj[:, -1].float()
        sl = torch.tensor(d["student_sigmas"], device=dev)
        nblk = (F_ + a.block - 1) // a.block
        ib = torch.randint(0, len(sl), (B, nblk), generator=g).to(dev)
        idx = ib.repeat_interleave(a.block, dim=1)[:, :F_]
        z = torch.gather(traj, 1, idx.view(B,1,1,F_,1,1).expand(B,1,C,F_,H,W)).squeeze(1)
        sig = sl[idx]
        rw.clear(); refu(d["ref_latent"].to(dev, dt), torch.zeros((), device=dev).long(),
                         encoder_hidden_states=d["clip_emb"].to(dev, dt), return_dict=False)
        rr.update(rw, dtype=dt)
        v = denu(z, (sig*1000.0).to(dt), encoder_hidden_states=[d["clip_emb"].to(dev,dt),
                 d["motion"].to(dev,dt)], pose_cond_fea=None, return_dict=False)[0]
        x0 = z.float() - sig.view(B,1,F_,1,1)*v.float()
        acc.append(float(((x0-target)**2).mean()))
    return float(np.mean(acc))

print(f"\n{'ckpt':>26}{'训练集':>11}{'held-out':>11}{'gap':>9}{'gap%':>8}")
print("-"*66)
prev = None
for ck in a.ckpts:
    k = torch.load(ck, map_location="cpu")
    denu.load_state_dict(k.get("denoising_unet", k), strict=False); denu.eval()
    tr, ho = loss_on(sets["训练集"]), loss_on(sets["held-out"])
    nm = os.path.basename(ck)[:24]
    print(f"{nm:>26}{tr:>11.5f}{ho:>11.5f}{ho-tr:>9.5f}{(ho-tr)/tr*100:>7.1f}%", flush=True)
print(f"""
判读:
  · held-out 随训练**上升** → 过拟合,多生成轨迹有用
  · held-out **下降**而 GT 验证升 → 在逼近不完美 teacher,加数据无用(天花板在 teacher)
  · gap% 随训练**扩大** → 泛化在恶化(过拟合的直接证据)""")

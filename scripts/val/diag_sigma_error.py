"""σ 分辨的 teacher 误差诊断:判断"训练 σ 采样分布是否欠采样高噪段"。

背景
----
stage2 训练用 σ ~ sigmoid(N(0,1))(logit-normal,无 shift):中位数 0.499,
σ>0.75 只占 13.5%,σ>0.9 只占 1.3%。而推理侧最优网格是 shift3 = [1,.9,.75,.5],
**一半的推理步落在 σ>0.75**。若训练确实欠采样高噪,该段误差应显著偏高。

关键判据不是"高 σ 误差绝对值大"(那是必然的,信息本来就少),而是
**不同训练步数的 ckpt 之间,改善集中在哪一段**:
  · 改善集中在高 σ  → 高噪段还在收敛 = 欠采样,给训练加 shift 有理
  · 改善均匀/集中在低 σ → 高噪段已收敛,加 shift 只会挤占已收敛区的样本

用 held-out 测试集(testset,不在训练集内),固定噪声种子,逐 σ 算 v-MSE。
cfg=1(纯条件分支),测的是模型本身而非采样配方。

用法:
  python scripts/val/diag_sigma_error.py --gpu 6 --clips 12 \
      --ckpts output/flow_stage2_cfgdrop/CUM1000.pt output/flow_stage2_cfgdrop/CUM1500.pt \
              output/flow_stage2_cfgdrop/CUM2000.pt
"""
import os, sys, json, argparse
import numpy as np, torch
XN = "/media/ps/ssd5/ayr/x-nemo-inference"
for p in (XN, "/media/ps/ssd5/ayr/motar"):
    if p not in sys.path: sys.path.append(p)
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

TESTSET = "/media/ps/ssd5/ayr/eval_metrics/testset"
POSE = "/media/ps/ssd4/ayr/hallo3_frames_512/pose_embed"
LAT  = "/media/ps/ssd4/ayr/hallo3_frames_512/frame_latent"

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=6)
ap.add_argument("--clips", type=int, default=12)
ap.add_argument("--L", type=int, default=64, help="与训练一致的片段长度")
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--sigmas", nargs="+", type=float,
                default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
ap.add_argument("--uncond", action="store_true",
                help="按 XNeMo 三路置空测 **uncond 分支**(bank不注入+CLIP置零+参考帧motion),"
                     "与 flow_stage2_temporal.py:201-204 逐条一致。"
                     "★ 三路置空全部作用在**空间主干**上,而 stage2 冻结主干 —— "
                     "若 uncond v-MSE 随 stage1 训练步数上升,就是纯 cond 微调在侵蚀 uncond 分支")
ap.add_argument("--no_temporal", action="store_true",
                help="测 stage1(空间主干)。按 flow_stage1_image.py 的做法彻底移除 temporal "
                     "self-attn(use_temporal_module=False),否则会把 ε 空间的旧 temporal 混进来")
ap.add_argument("--out", default=f"{XN}/output/diag_sigma_error.json")
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); dt = torch.bfloat16
torch.cuda.set_device(dev)

cfg = OmegaConf.load(f"{XN}/configs/test_ar_model.yaml"); ic = OmegaConf.load(cfg.inference_config)
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
uak = OmegaConf.to_container(ic.unet_additional_kwargs, resolve=True)
if a.no_temporal:
    uak["use_temporal_module"] = False          # 与 flow_stage1_image.py 一致
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=uak).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
if not a.no_temporal:
    denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    set_temporal_rope(denu, True, mode="bidir")
else:
    n_tmp = sum(1 for n, _ in denu.named_parameters() if "temporal_modules" in n)
    assert n_tmp == 0, f"temporal_modules 未被移除,仍有 {n_tmp} 个"
    print("[mode] 空间主干(temporal 已移除)", flush=True)
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read",  batch_size=1, fusion_blocks="full")

man = json.load(open(f"{TESTSET}/manifest.json"))
sel = json.load(open(f"{TESTSET}/hallo3_subset30.json"))
clips = [c for c in man["hallo3"] if c in set(sel)][:a.clips]
print(f"[data] {len(clips)} 条 held-out clip(hallo3 subset30), L={a.L}", flush=True)

# 预载数据 + 固定噪声(所有 ckpt / 所有 σ 共用同一份噪声,消除采样方差)
items = []
for c in clips:
    m = man["hallo3"][c]
    if m["n_frames"] < a.L: continue
    lat = torch.load(f"{LAT}/{c}.pt", map_location="cpu").float()[:a.L]
    mo = torch.load(f"{POSE}/{c}.pt", map_location="cpu").float().reshape(-1, 32*16)[:a.L]
    ref_pil = Image.open(m["ref"]).convert("RGB").resize((512, 512))
    with torch.no_grad():
        ce = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)),
             return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    x0 = lat.permute(1,0,2,3).unsqueeze(0).to(dev, dt)          # [1,4,L,64,64]
    g = torch.Generator(device=dev); g.manual_seed(1234)
    eps = torch.randn(x0.shape, generator=g, device=dev, dtype=torch.float32)
    items.append((c, x0, lat[0:1].to(dev, dt), ce, mo.reshape(1,a.L,32,16).to(dev,dt), eps))
print(f"[data] 实际可用 {len(items)} 条", flush=True)

@torch.no_grad()
def vmse(ck):
    k = torch.load(ck, map_location="cpu")
    denu.load_state_dict(k["denoising_unet"], strict=False); denu.eval()
    out = {}
    for s in a.sigmas:
        tot, n = 0.0, 0
        for c, x0, ref, ce, mo, eps in items:
            rw.clear(); refu(ref, torch.zeros((), device=dev).long(),
                             encoder_hidden_states=ce, return_dict=False)
            rr.update(rw, dtype=dt)
            ce_i, mo_i = ce, mo
            if a.uncond:                       # 与 stage2 的 cfg_drop 分支逐条一致
                rr.clear()                                        # ① bank 不注入
                ce_i = torch.zeros_like(ce)                       # ② CLIP 置零
                mo_i = mo[:, 0:1].expand_as(mo).contiguous()      # ③ 参考帧 motion
            x0f = x0.float()
            z = ((1-s)*x0f + s*eps).to(dt)
            v = denu(z, torch.full((1,), s*1000.0, device=dev, dtype=dt),
                     encoder_hidden_states=[ce_i, mo_i], pose_cond_fea=None, return_dict=False)[0]
            tot += float(((v.float() - (eps - x0f))**2).mean()); n += 1
        out[s] = tot/n
        print(f"    σ={s:<5} v_mse={out[s]:.5f}", flush=True)
    return out

def _lab(ck):
    """★ 必须带父目录:flow_stage1/ 与 flow_stage1_shift3/ 下的 ckpt 同名,
    只用 basename 会互相覆盖(曾因此产出一张前后半段完全重复的表)。"""
    d = os.path.basename(os.path.dirname(ck))
    return f"{d.replace('flow_stage1','s1')}/{os.path.basename(ck)[len('stage1_step_'):-3]}" \
        if "stage1_step_" in ck else f"{d}/{os.path.basename(ck)[:-3]}"

res = {}
for ck in a.ckpts:
    print(f"\n[ckpt] {ck}", flush=True)
    res[_lab(ck)] = vmse(ck)
json.dump(res, open(a.out, "w"), indent=2)

# ---- 判读表:相邻 ckpt 的逐 σ 改善
names = [_lab(c) for c in a.ckpts]
print(f"\n{'σ':>6}" + "".join(f"{n[:13]:>15}" for n in names) +
      (f"{'改善%':>10}" if len(names)>1 else ""))
print("-"*(6+14*len(names)+10))
for s in a.sigmas:
    row = f"{s:>6}" + "".join(f"{res[n][s]:>15.5f}" for n in names)
    if len(names) > 1:
        imp = (res[names[0]][s]-res[names[-1]][s])/res[names[0]][s]*100
        row += f"{imp:>9.2f}%"
    print(row)
print(f"\n判读:'改善%' 若随 σ 单调上升 → 高噪段仍在收敛 = 训练欠采样高噪,加 shift 有理;"
      f"\n     若在低/中 σ 更大或无趋势 → 高噪段已收敛,加 shift 只会挤占已收敛区")
print(f"\n写入 {a.out}")

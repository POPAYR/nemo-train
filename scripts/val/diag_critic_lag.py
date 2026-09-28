"""区分 C loss 上升的两种成因:critic 追踪滞后 vs 生成器分布内在变难。

背景:DMD 训练中 C loss 单调爬升(0.19→0.54 后崩)。DMD2 把不稳定归因于
"fake critic 估不准生成分布"(追踪滞后)。但 C loss 上升还有另一个可能:
生成器 mode collapse 后分布变尖锐,**尖锐分布本身就更难用平滑网络拟合**。
**两者处方相反**:前者要加强 critic(提 lr / 提 ratio / 降 gen_lr),
后者要阻止坍缩(分段 CFG / 正则)。不区分就是盲调 —— 我们调 ratio 失败很可能就是押错边。

协议(公平对照):对每个**冻结的** generator,critic **一律从 teacher 权重重新开始**,
只训 critic N 步,看 C loss 能收敛到多低。
  · 能降到低位 → 该分布可拟合,联合训练时的高 C loss 是**滞后**
  · 降不下去(高位平台) → 该分布**内在难拟合**,是坍缩的后果

对照组(按 DMD 训练进程排列):
  ode_init      未经 DMD,分布最"宽"                  → 预期最容易
  dmd_2000      DMD 峰值 ckpt(交付物)
  v6_4000       分段CFG 训到累计4000(DYN 1.096,疑退化)

用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/diag_critic_lag.py --steps 200
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from data.dataset import MotarDataset
from src.distill.models import DMD2Models
from src.distill.flow_step import critic_loss as flow_critic_loss

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/critic_lag"
CKPTS = {
    "ode_init":  ("output/ode_init_causal/odeinit_step_2500.pt", "denoising_unet"),
    "dmd_2000":  ("output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt", "generator_ema"),
    "v6_4000":   ("output/flowdmd_segcfg_v6/ckpt/dmd2_step_2000.pt", "generator_ema"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--gens", nargs="+", default=list(CKPTS))
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--critic_lr", type=float, default=2e-6, help="诊断用,比训练的 4e-7 高 5×以在有限步内看到收敛趋势")
ap.add_argument("--L", type=int, default=32)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--window", type=int, default=24)
ap.add_argument("--sigma_high", type=float, default=0.98)
ap.add_argument("--log_every", type=int, default=20)
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16
DSL = [1.0, 0.75, 0.5, 0.25]

dcfg = OmegaConf.load(TRAIN_CFG).data
vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
ds = ConcatDataset([MotarDataset(
        pose_dir=s.pose_dir, audio_dir=s.audio_dir, caption_dir=s.caption_dir,
        data_name_path=s.data_name_path, tokenizer_path=dcfg.tokenizer_path,
        data_stats_path=dcfg.data_stats_path, context_length=a.L, fps=dcfg.fps, sr=dcfg.sr,
        text_max_len=dcfg.get("text_max_len", 128), random_crop=True, pad_short=True,
        load_video=True, latent_dir=s.latent_dir, video_dir=s.video_dir, video_processor=vproc)
    for s in dcfg.sources])
stats = torch.load(dcfg.data_stats_path, map_location="cpu")
m_mean = stats["mean"].reshape(-1).to(dev, dt); m_std = stats["std"].reshape(-1).to(dev, dt)
print(f"[data] {len(ds)} clips", flush=True)

results = {}
for name in a.gens:
    ck, key = CKPTS[name]
    print(f"\n{'='*70}\n[gen] {name}  ← {ck}  key={key}", flush=True)
    # ★ critic 一律从 teacher 权重起步(DMD2Models 构造时 critic←flow_ckpt),保证公平
    M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=a.block, objective="flow",
                   flow_ckpt="output/flow_stage2_cfgdrop/CUM1500.pt")
    sd = torch.load(ck, map_location="cpu")
    M.generator.load_state_dict(sd[key], strict=False)
    M.generator.eval().requires_grad_(False)          # ★ 冻结 generator
    M.gen_causal.set_mode("train")
    cp = [p for p in M.critic.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(cp, lr=a.critic_lr, betas=(0.0, 0.999), weight_decay=0.01)

    g = torch.Generator(); g.manual_seed(1234)        # 三组用同一串数据,消除数据差异
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                    drop_last=True, generator=g)
    it = iter(dl); curve = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()
        B, T = x0.shape[0], x0.shape[2]
        motion = (b["motion_tensor"].to(dev, dt) * (m_std + 1e-6) + m_mean).reshape(B, T, 32, 16)
        with torch.no_grad():
            clip = M.clip_embed(b["ref_img"].to(dev, dt))
            M.set_reference(b["ref_latent"].to(dev, dt), clip, B)
        noise = torch.randn(B, 4, T, 64, 64, device=dev, dtype=dt)
        opt.zero_grad(set_to_none=True)
        lc, logc = flow_critic_loss(M, noise, clip, motion, DSL, block_size=a.block,
                                    grad_window=a.window, sigma_high=a.sigma_high)
        lc.backward()
        torch.nn.utils.clip_grad_norm_(cp, 10.0)
        opt.step()
        curve.append(float(lc.item()))
        if step % a.log_every == 0:
            w = curve[-a.log_every:]
            print(f"  step {step:4d}  C={np.mean(w):.4f}  (首{a.log_every}步均值 "
                  f"{np.mean(curve[:a.log_every]):.4f})  {(time.time()-t0)/step:.1f}s/it", flush=True)
    first = float(np.mean(curve[:a.log_every]))
    last = float(np.mean(curve[-a.log_every:]))
    results[name] = {"first": first, "last": last, "drop_pct": (first - last) / first * 100,
                     "curve": curve}
    print(f"[{name}] 首{a.log_every}步 {first:.4f} → 末{a.log_every}步 {last:.4f} "
          f"(降 {results[name]['drop_pct']:.1f}%)", flush=True)
    del M, opt, cp; torch.cuda.empty_cache()

json.dump(results, open(f"{OUT}/critic_lag.json", "w"), indent=2)
print(f"\n{'='*70}\n{'generator':<14}{'首段C':>10}{'末段C':>10}{'降幅':>9}   判读")
print("-" * 70)
base = None
for n in a.gens:
    r = results[n]
    print(f"{n:<14}{r['first']:>10.4f}{r['last']:>10.4f}{r['drop_pct']:>8.1f}%")
print("\n判读指南:")
print("  · 各 generator 的**末段 C** 相近 → 分布难度相当,联合训练的高 C 是**追踪滞后**")
print("    → 处方:加强 critic(提 critic_lr / 提 ratio)或降 gen_lr")
print("  · 越靠后的 ckpt 末段 C 越高且降幅越小 → 分布**内在变难**(坍缩后果)")
print("    → 处方:阻止坍缩(分段CFG / ODE轨迹回归正则),加强 critic 无用")
print("DONE", flush=True)

"""显存探测:L=64 / window=64 的 DMD generator 步能否装下。

背景:`window=24` 的原始理由是"teacher PE max 32"——那是**旧的绝对 PE teacher** 的限制。
现 teacher = CUM1500(RoPE,在 L=64 上训),该上限已不存在。L=64 若仍用 window=24,
梯度只覆盖 37.5% 的帧(官方是全帧梯度),浪费。

⚠️ 本探测在**单卡 + 不开 ZeRO** 下跑,是**悲观估计**:
真实训练是 4 卡 ZeRO,Adam 优化器态分片到 1/4。
故"单卡装得下" ⇒ 4卡ZeRO 必然装得下;"单卡 OOM" 则需按下方公式折算再判断。

用法: CUDA_VISIBLE_DEVICES=4 python scripts/val/probe_rollout_mem.py
"""
import os, sys, argparse, gc
import torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from src.distill.models import DMD2Models
from src.distill.flow_step import generator_loss as flow_generator_loss

ap = argparse.ArgumentParser()
ap.add_argument("--combos", nargs="+", default=["32:24", "64:24", "64:48", "64:64"],
                help="L:window 组合")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--guidance", type=float, default=2.5)
ap.add_argument("--flow_ckpt", default="output/flow_stage2_cfgdrop/CUM1500.pt")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
DSL = [1.0, 0.75, 0.5, 0.25]
TOT = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[gpu] {torch.cuda.get_device_name(0)}  总显存 {TOT:.1f}GB", flush=True)

M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=a.block, objective="flow",
               flow_ckpt=a.flow_ckpt)
gp = [p for p in M.generator.parameters() if p.requires_grad]
cp = [p for p in M.critic.parameters() if p.requires_grad]
# 真实训练有 gen+critic 两个 AdamW;这里只建 gen 的(critic 步是另一次前向,峰值不叠加)
opt = torch.optim.AdamW(gp, lr=2e-6, betas=(0.0, 0.999), weight_decay=0.01)
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated() / 1e9
print(f"[base] 模型+优化器(未含激活) {base:.1f}GB", flush=True)

print(f"\n{'L':>4}{'window':>8}{'梯度覆盖':>10}{'峰值显存':>10}   结果")
print("-" * 56)
for combo in a.combos:
    L, W = map(int, combo.split(":"))
    try:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats()
        clip = torch.randn(1, 1, 768, device=dev, dtype=dt)
        motion = torch.randn(1, L, 32, 16, device=dev, dtype=dt)
        ref = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)
        M.set_reference(ref, clip, 1)
        noise = torch.randn(1, 4, L, 64, 64, device=dev, dtype=dt)
        opt.zero_grad(set_to_none=True)
        lg, _ = flow_generator_loss(M, noise, clip, motion, DSL, block_size=a.block,
                                    grad_window=W, guidance_scale=a.guidance, sigma_high=0.98)
        lg.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{L:>4}{W:>8}{W/L*100:>9.0f}%{peak:>9.1f}GB   ✅ 可行", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{L:>4}{W:>8}{W/L*100:>9.0f}%{'—':>10}   ❌ OOM(单卡无ZeRO)", flush=True)
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"{L:>4}{W:>8}{'—':>10}{'—':>10}   ⚠️ {type(e).__name__}: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache(); gc.collect()

print(f"""
折算到真实训练(4卡 ZeRO):
  · Adam 态 = 2模型 × 1.68B × (m,v) × 4B ≈ 27GB,ZeRO 分片后每卡约 6.7GB(省约 20GB)
  · 但真实训练同时持有 gen+critic 两套梯度,本探测只有 gen 一套(多约 3.4GB)
  · 净效应:真实每卡 ≈ 本探测峰值 − 约 17GB
  · 参考实测:L=32/window=24 真实训练为 44.3GB/卡""")
print("DONE", flush=True)

"""
Path A / 方案 C 诊断：torch.compile 到底是「按 shape」还是「按调用」recompile？
==============================================================================
借鉴 Self-Forcing 官方 real-time 配方：官方用 `mode="max-autotune-no-cudagraphs"`
（不用 CUDA-graph！）+ FlexAttention(compile 友好) + 预分配静态 KV-cache → 只在首块编译一次。
我们上一版 bench 用的是 reduce-overhead(CUDA-graph 模式)= 最挑架构的那个。

本诊断：
  - cache_size_limit 抬到 256 + 打开 recompile 日志（看 recompile 原因）。
  - 用官方同款 mode="max-autotune-no-cudagraphs"，dynamic=False。
  - 对 **固定 F=8、固定输入** 连打 N 次前向：
      · 若首次编译后不再 recompile → 「按 shape」，抬 limit 即可，白捡加速。
      · 若每次都 recompile → 「按调用」(bank/einops 动态状态)，抬 limit 无效，必须走架构改造。
  - 报告：编译后稳态延迟 vs eager。

用法（可与训练并存，跑在别的卡上；计时受邻居干扰但 recompile 计数是确定的）：
  CUDA_VISIBLE_DEVICES=1 python scripts/val/diag_compile_recompile.py
"""
import os
import sys
import time
import numpy as np
import torch

XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path:
    sys.path.append(XNEMO_ROOT)

from omegaconf import OmegaConf
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder

# 抬高 recompile 上限 + 打开 recompile 归因日志
torch._dynamo.config.cache_size_limit = 256
for _attr in ("accumulated_cache_size_limit",):   # 新版才有，老版忽略
    if hasattr(torch._dynamo.config, _attr):
        setattr(torch._dynamo.config, _attr, 512)
try:
    torch._logging.set_logs(recompiles=True, graph_breaks=True)
except Exception as e:
    print("[warn] set_logs:", e)


def main():
    device = torch.device("cuda:0")
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[env] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"mode=max-autotune-no-cudagraphs  cache_limit={torch._dynamo.config.cache_size_limit}")

    config = OmegaConf.load(os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    print("[load] building decoder ...")
    ref_unet, den_unet, _ = build_decoder(config, device, dtype)

    H = W = 64
    F = 8
    writer = ReferenceAttentionControl(ref_unet, do_classifier_free_guidance=False,
                                       mode="write", batch_size=1, fusion_blocks="full")
    reader = ReferenceAttentionControl(den_unet, do_classifier_free_guidance=False,
                                       mode="read", batch_size=1, fusion_blocks="full")
    clip_emb = torch.randn(1, 1, 768, device=device, dtype=dtype)
    ref_latent = torch.randn(1, 4, H, W, device=device, dtype=dtype)
    t = torch.tensor(500, device=device)
    writer.clear()
    ref_unet(ref_latent, torch.zeros_like(t), encoder_hidden_states=clip_emb, return_dict=False)
    reader.update(writer)   # bank 一次性静态化（同真实推理）

    # 固定输入张量（每次调用完全相同 → 若还 recompile 就是「按调用」的动态状态）
    lat = torch.randn(1, 4, F, H, W, device=device, dtype=dtype)
    mot = torch.randn(1, F, 32, 16, device=device, dtype=dtype)

    def run(net):
        return net(lat, t, encoder_hidden_states=[clip_emb, mot], pose_cond_fea=None, return_dict=False)

    # ---- eager 稳态 ----
    for _ in range(3): run(den_unet)
    torch.cuda.synchronize()
    ts = []
    for _ in range(12):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(den_unet); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    eager_med = float(np.median(ts))
    print(f"\n[eager] F={F} median={eager_med:.2f} ms")

    # ---- compiled：mode=default（编译快，先看能否收敛；收敛再考虑 max-autotune 榨性能）----
    import os as _os
    _mode = _os.environ.get("COMPILE_MODE", "default")
    print(f"\n[compile] mode={_mode}, dynamic=False, cache_limit={torch._dynamo.config.cache_size_limit}")
    print("  ~77 个子模块实例 → 按 obj_id 各编译一次(一次性)；跑够多次让它编完、看稳态。")
    den_c = torch.compile(den_unet, mode=_mode, fullgraph=False, dynamic=False)

    # 多跑几轮，让所有实例编完 → 观察是否收敛到稳态快速
    print("[compiled] 逐次调用延迟（前几次含编译会慢；若最终收敛=一次性，若永远抖=真按调用）：")
    per_call = []
    for i in range(30):
        torch.cuda.synchronize(); st = time.time()
        run(den_c)
        torch.cuda.synchronize(); dt = (time.time() - st) * 1000
        per_call.append(dt)
        if i < 8 or i % 5 == 0 or i >= 27:
            rc = 0
            print(f"   call {i:2d}: {dt:8.1f} ms")
    # 稳态取后 8 次
    steady = np.median(per_call[-8:])
    print(f"\n[compiled] 稳态 median(后8次)={steady:.2f} ms   vs eager {eager_med:.2f} ms  → ×{eager_med/steady:.2f}")
    # 判定：后 8 次是否稳定（std 小=收敛到编译版；大=还在反复 recompile）
    tail_std = float(np.std(per_call[-8:]))
    verdict = ("收敛（一次性编译，抬 limit 有效！可走 compile）" if tail_std < 0.15 * steady
               else "未收敛（后段仍抖=真按调用 → 需架构改造）")
    print(f"[判定] 后8次 std={tail_std:.1f} ms（{100*tail_std/steady:.0f}%）→ {verdict}")
    print(f"[mem] 峰值 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("\n注：上方 dynamo 日志里 'Recompiling function ...' 的次数是权威依据；per-call 计时是旁证。")


if __name__ == "__main__":
    main()

"""
Fullgraph 后的实时提速实测：eager vs compile(default) vs compile(reduce-overhead=CUDA-graph)
============================================================================================
前提：改 `.sample`→`return_dict=False)[0]` 后 UNet 已是 **1 graph / 0 break**（diag_graph_breaks 实测）。
现在 compile 能真正融合 + CUDA-graph 能干净捕获 → 量真实提速与 2/4 步 fps。

用法：CUDA_VISIBLE_DEVICES=0 python scripts/val/bench_fullgraph.py
"""
import os, sys, time
import numpy as np
import torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder
torch._dynamo.config.cache_size_limit = 256
TAESD = "/media/ps/ssd5/ayr/pretrained/taesd"


def bench(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    dev = torch.device("cuda:0"); dt = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    print(f"[env] torch={torch.__version__}  {torch.cuda.get_device_name(0)}")
    ru, du, _ = build_decoder(cfg, dev, dt)
    vae = AutoencoderTiny.from_pretrained(TAESD, torch_dtype=dt).to(dev).eval()

    H = W = 64; F = 8
    w = ReferenceAttentionControl(ru, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
    r = ReferenceAttentionControl(du, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
    ce = torch.randn(1, 1, 768, device=dev, dtype=dt); rl = torch.randn(1, 4, H, W, device=dev, dtype=dt); t = torch.tensor(500, device=dev)
    w.clear(); ru(rl, torch.zeros_like(t), encoder_hidden_states=ce, return_dict=False); r.update(w)
    lat = torch.randn(1, 4, F, H, W, device=dev, dtype=dt); mot = torch.randn(1, F, 32, 16, device=dev, dtype=dt)
    z = torch.randn(F, 4, H, W, device=dev, dtype=dt)

    def unet_eager(): return du(lat, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)
    def vae_dec(): return vae.decode(z).sample

    t_eager = bench(unet_eager, 5, 20)
    t_vae = bench(vae_dec, 3, 10)
    print(f"\n[eager]  unet(F=8) {t_eager:.2f} ms   taesd {t_vae:.2f} ms")

    results = {"eager": t_eager}
    for mode in ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]:
        torch._dynamo.reset()
        try:
            duc = torch.compile(du, mode=mode, fullgraph=True, dynamic=False)
            def fn(): return duc(lat, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)
            print(f"[compile:{mode}] fullgraph=True 编译中（首次慢）...")
            tt = bench(fn, 12, 20)   # 多 warmup 让编译/CUDA-graph 稳定
            results[mode] = tt
            print(f"[compile:{mode}]  unet {tt:.2f} ms   ×{t_eager/tt:.2f}")
        except Exception as ex:
            print(f"[compile:{mode}] 失败: {str(ex)[:200]}")
            results[mode] = float("nan")

    # fps 表
    best = min([v for k, v in results.items() if k != "eager" and v == v], default=t_eager)
    print(f"\n{'='*60}\n[fps]  block=8  fps = block/(S·t_unet + t_vae)   目标 25\n{'='*60}")
    for tag, tu in [("eager", t_eager), ("best-compiled", best)]:
        for S in [2, 4]:
            tot = S*tu + t_vae
            print(f"  {tag:14s} S={S}: {1000*F/(S*tu):6.1f} fps(unet-only)  {1000*F/tot:6.1f} fps(+vae)")
    print(f"\n  峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()

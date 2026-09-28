"""fp16 vs bf16 去噪 UNet 前向延迟对比（block=8）。理论上同 Tensor Core 速率 → 应无差。"""
import os, sys
import numpy as np, torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder


def bench(fn, warmup=8, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def run(dt, cfg, dev):
    ru, du, _ = build_decoder(cfg, dev, dt); du.eval()
    H = W = 64; F = 8
    w = ReferenceAttentionControl(ru, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
    r = ReferenceAttentionControl(du, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
    ce = torch.randn(1,1,768,device=dev,dtype=dt); rl = torch.randn(1,4,H,W,device=dev,dtype=dt); t = torch.tensor(500,device=dev)
    w.clear(); ru(rl, torch.zeros_like(t), encoder_hidden_states=ce, return_dict=False); r.update(w)
    lat = torch.randn(1,4,F,H,W,device=dev,dtype=dt); mot = torch.randn(1,F,32,16,device=dev,dtype=dt)
    def fn(): return du(lat, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)
    m = bench(fn)
    del ru, du, w, r; torch.cuda.empty_cache()
    return m


def main():
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    print(f"[env] {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    m16 = run(torch.float16, cfg, dev)
    mbf = run(torch.bfloat16, cfg, dev)
    print(f"\n  fp16  unet(F=8) = {m16:.2f} ms")
    print(f"  bf16  unet(F=8) = {mbf:.2f} ms")
    print(f"  bf16/fp16 = {mbf/m16:.3f}  →  {'基本无差(同 TC 速率)' if abs(mbf/m16-1)<0.05 else '有差异'}")


if __name__ == "__main__":
    main()

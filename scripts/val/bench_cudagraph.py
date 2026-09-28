"""Fullgraph 后：手动 CUDA Graph 实测 launch 开销消除（部署态最快路径，秒级 setup）。"""
import os, sys
import numpy as np, torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder


def bench(fn, warmup=8, iters=30):
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
    print(f"[env] torch={torch.__version__} {torch.cuda.get_device_name(0)}")
    ru, du, _ = build_decoder(cfg, dev, dt); du.eval()
    vae = AutoencoderTiny.from_pretrained("/media/ps/ssd5/ayr/pretrained/taesd", torch_dtype=dt).to(dev).eval()
    H = W = 64; F = 8
    w = ReferenceAttentionControl(ru, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
    r = ReferenceAttentionControl(du, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
    ce = torch.randn(1, 1, 768, device=dev, dtype=dt); rl = torch.randn(1, 4, H, W, device=dev, dtype=dt); t = torch.tensor(500, device=dev)
    w.clear(); ru(rl, torch.zeros_like(t), encoder_hidden_states=ce, return_dict=False); r.update(w)
    lat = torch.randn(1, 4, F, H, W, device=dev, dtype=dt); mot = torch.randn(1, F, 32, 16, device=dev, dtype=dt)
    z = torch.randn(F, 4, H, W, device=dev, dtype=dt)

    def unet_eager(): return du(lat, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)
    t_eager = bench(unet_eager); t_vae = bench(lambda: vae.decode(z).sample, 3, 10)
    print(f"\n[eager] unet {t_eager:.2f} ms  taesd {t_vae:.2f} ms")

    # ---- 手动 CUDA Graph 捕获 UNet ----
    t_cg = float("nan")
    try:
        with torch.no_grad():
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): unet_eager()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = du(lat, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)
        t_cg = bench(lambda: g.replay())
        print(f"[cudagraph] unet {t_cg:.2f} ms  ×{t_eager/t_cg:.2f}")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[cudagraph] 失败: {str(ex)[:300]}")

    print(f"\n{'='*56}\n[fps] block=8  目标25\n{'='*56}")
    for tag, tu in [("eager", t_eager), ("cudagraph", t_cg)]:
        if tu != tu: continue
        for S in [2, 4]:
            print(f"  {tag:10s} S={S}: {1000*F/(S*tu):6.1f} fps(unet)  {1000*F/(S*tu+t_vae):6.1f} fps(+vae)")
    print(f"\n峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()

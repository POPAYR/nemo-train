"""
Path A 实时基准：4 步 + torch.compile / CUDA-Graph 能否把因果解码器推到 25fps
==========================================================================
背景（DECODER_DISTILL_PLAN.md §Phase0 / decoder-distill-project）：我们的负载是
launch/latency-bound（A100 只吃 ~70/250W），裸 FLOPS 没吃满 → torch.compile(reduce-overhead,
用 CUDA-Graph 把 kernel launch 开销摊掉) 有望 ~2-3×。本脚本量「eager vs compiled」下
去噪 UNet(block=8) + VAE decode 的逐块延迟，推 2 步/4 步稳态 fps，判断保底实时是否成立。

测法：
  - 复用 phase0 的 build_decoder（reference + 3D 去噪 UNet + VAE），部署态 batch=1 无 CFG。
  - VAE 默认 TAESD（AutoencoderTiny，实时交付解码器；SVD 是质量解码器，另测）。
  - den_unet 前向 @ F=block（默认 8），VAE decode @ F=block：分别 eager 与 compiled 计时。
  - fps 推导：稳态每 block 耗时 = S·t_unet(block) + t_vae(block)，fps = block / 耗时。
    （UNet 与下一 block 的 VAE 可流水并行，此为不并行的保守下界。）
  - reference-attention 用 python hook 读写 bank → 可能 graph break：compile 用 fullgraph=False
    容错，失败则回退 eager 并打印原因（本身就是有价值的结论：compile 是否兼容我们的架构）。

用法：
  CUDA_VISIBLE_DEVICES=2 python scripts/val/bench_decoder_compile.py --block 8 --vae taesd
  # 可选 --vae svd（质量解码器对照）；--compile_mode reduce-overhead|max-autotune|default
"""
import argparse
import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf

XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path:
    sys.path.append(XNEMO_ROOT)

from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder, timeit, n_params

TAESD_PATH = "/media/ps/ssd5/ayr/pretrained/taesd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    ap.add_argument("--block", type=int, default=8, help="因果 block 帧数（我们的 block_size）")
    ap.add_argument("--extra_frames", default="4,24", help="额外对照的 F（逗号）")
    ap.add_argument("--vae", default="taesd", choices=["taesd", "svd"], help="解码器：taesd=实时/svd=质量")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--compile_mode", default="reduce-overhead",
                    choices=["reduce-overhead", "max-autotune", "default"])
    ap.add_argument("--steps", default="2,4", help="去噪步数（逗号）用于 fps 推导")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[env] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"dtype={args.dtype}  compile_mode={args.compile_mode}  vae={args.vae}")

    config = OmegaConf.load(args.config)
    print("[load] building decoder ...")
    ref_unet, den_unet, svd_vae = build_decoder(config, device, dtype)
    print(f"[params] denoising_unet={n_params(den_unet)/1e6:.1f}M")

    # ---- VAE 选择 ----
    if args.vae == "taesd":
        from diffusers import AutoencoderTiny
        vae = AutoencoderTiny.from_pretrained(TAESD_PATH, torch_dtype=dtype).to(device).eval()
        def vae_decode(z):  # z:[F,4,64,64]
            return vae.decode(z).sample
    else:
        vae = svd_vae
        def vae_decode(z):
            return vae.decode(z, z.shape[0]).sample

    H = W = 64
    block = args.block
    F_list = sorted(set([block] + [int(x) for x in args.extra_frames.split(",") if x]))

    # ---- reference bank（部署态 batch=1，整片一次，摊销≈0）----
    writer = ReferenceAttentionControl(ref_unet, do_classifier_free_guidance=False,
                                       mode="write", batch_size=1, fusion_blocks="full")
    reader = ReferenceAttentionControl(den_unet, do_classifier_free_guidance=False,
                                       mode="read", batch_size=1, fusion_blocks="full")
    clip_emb = torch.randn(1, 1, 768, device=device, dtype=dtype)
    ref_latent = torch.randn(1, 4, H, W, device=device, dtype=dtype)
    t = torch.tensor(500, device=device)
    writer.clear()
    ref_unet(ref_latent, torch.zeros_like(t), encoder_hidden_states=clip_emb, return_dict=False)
    reader.update(writer)

    def make_unet_fn(F):
        lat = torch.randn(1, 4, F, H, W, device=device, dtype=dtype)
        mot = torch.randn(1, F, 32, 16, device=device, dtype=dtype)
        def fn():
            return den_unet(lat, t, encoder_hidden_states=[clip_emb, mot],
                            pose_cond_fea=None, return_dict=False)
        return fn

    def make_vae_fn(F):
        z = torch.randn(F, 4, H, W, device=device, dtype=dtype)
        def fn():
            return vae_decode(z)
        return fn

    # ---- 1) eager 计时 ----
    print(f"\n{'='*66}\n[EAGER]  den_unet + {args.vae} decode  (batch=1, no CFG)\n{'='*66}")
    eager_unet, eager_vae = {}, {}
    for F in F_list:
        m, _, _ = timeit(make_unet_fn(F), warmup=5, iters=15)
        eager_unet[F] = m
        try:
            mv, _, _ = timeit(make_vae_fn(F), warmup=3, iters=10); eager_vae[F] = mv
        except Exception as ex:
            eager_vae[F] = float("nan"); print(f"  [vae F={F}] err {ex}")
        print(f"  F={F:>3}  unet {m:>7.2f} ms ({m/F:>5.2f}/帧)   vae {eager_vae[F]:>7.2f} ms")

    # ---- 2) torch.compile ----
    print(f"\n[compile] wrapping den_unet + vae (mode={args.compile_mode}, fullgraph=False) ...")
    comp_unet, comp_vae = {}, {}
    compiled_ok_unet = compiled_ok_vae = True
    try:
        den_c = torch.compile(den_unet, mode=args.compile_mode, fullgraph=False, dynamic=False)
    except Exception as ex:
        den_c = den_unet; compiled_ok_unet = False; print(f"  [unet compile ERR] {ex}")
    try:
        vae_c_mod = torch.compile(vae, mode=args.compile_mode, fullgraph=False, dynamic=False)
        if args.vae == "taesd":
            def vae_decode_c(z): return vae_c_mod.decode(z).sample
        else:
            def vae_decode_c(z): return vae_c_mod.decode(z, z.shape[0]).sample
    except Exception as ex:
        vae_decode_c = vae_decode; compiled_ok_vae = False; print(f"  [vae compile ERR] {ex}")

    def make_unet_fn_c(F):
        lat = torch.randn(1, 4, F, H, W, device=device, dtype=dtype)
        mot = torch.randn(1, F, 32, 16, device=device, dtype=dtype)
        def fn():
            return den_c(lat, t, encoder_hidden_states=[clip_emb, mot],
                         pose_cond_fea=None, return_dict=False)
        return fn

    def make_vae_fn_c(F):
        z = torch.randn(F, 4, H, W, device=device, dtype=dtype)
        def fn(): return vae_decode_c(z)
        return fn

    print(f"\n{'='*66}\n[COMPILED]  (首次含编译，warmup 会久)\n{'='*66}")
    for F in F_list:
        try:
            m, _, _ = timeit(make_unet_fn_c(F), warmup=8, iters=15); comp_unet[F] = m
        except Exception as ex:
            comp_unet[F] = float("nan"); print(f"  [unet-c F={F}] err {str(ex)[:120]}")
        try:
            mv, _, _ = timeit(make_vae_fn_c(F), warmup=5, iters=10); comp_vae[F] = mv
        except Exception as ex:
            comp_vae[F] = float("nan"); print(f"  [vae-c F={F}] err {str(ex)[:120]}")
        su = eager_unet[F]/comp_unet[F] if comp_unet[F] == comp_unet[F] and comp_unet[F] > 0 else float("nan")
        sv = eager_vae[F]/comp_vae[F] if comp_vae[F] == comp_vae[F] and comp_vae[F] > 0 else float("nan")
        print(f"  F={F:>3}  unet {comp_unet[F]:>7.2f} ms (×{su:.2f})   vae {comp_vae[F]:>7.2f} ms (×{sv:.2f})")

    # ---- 3) fps 表：稳态每 block 耗时 = S·t_unet(block) + t_vae(block) ----
    S_list = [int(x) for x in args.steps.split(",")]
    def fps_table(tu, tv, tag):
        print(f"\n[{tag} fps]  block={block}  fps = block/(S·t_unet + t_vae)   目标 25fps")
        for S in S_list:
            t_u = S * tu[block]
            t_full = t_u + (tv.get(block, 0.0) if tv.get(block, np.nan) == tv.get(block, np.nan) else 0.0)
            fps_u = 1000.0*block/t_u
            fps_f = 1000.0*block/t_full if t_full > 0 else float("nan")
            print(f"    S={S} 步:  unet-only {fps_u:>6.1f} fps   +vae {fps_f:>6.1f} fps   "
                  f"(每帧 {t_full/block:.2f} ms)")
    fps_table(eager_unet, eager_vae, "EAGER")
    if compiled_ok_unet:
        fps_table(comp_unet, comp_vae, "COMPILED")

    print(f"\n  峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("  说明：真实流式可把 VAE(上一 block) 与 UNet(下一 block) 流水并行 → 实际 fps 更接近 unet-only 列。")


if __name__ == "__main__":
    main()

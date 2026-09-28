"""
Phase-0 实时可行性闸门 benchmark (DECODER_DISTILL_PLAN.md §Phase 0)
================================================================
目的: 在写任何蒸馏代码前, 量出 X-Nemo 去噪解码器在单卡上的逐帧成本,
定死「步数 × block 大小 → fps」预算, 判断单 A100 @ 25fps 是否可达。

测什么:
  1. 去噪 UNet 单次 forward 延迟 @ 64x64 latent, 各 block 帧数 F ∈ {1,2,4,8,16,24}
     (batch=1 = student 无 CFG 部署态; 另测 batch=2 = teacher CFG 的开销对照)。
  2. reference UNet 单次 forward (整片只跑一次, 摊销成本)。
  3. SVD VAE decode 各 F 的延迟 (可能是瓶颈, 决定要不要换 TAEHV)。
  4. 推导 fps 表: 生成 K 帧耗时 = S * t_unet(K) + t_vae(K), fps = K / 耗时。

完全复用 test_ar_model.py 的模型构造 (含 reference-attention bank), 保证与真实推理一致。
推理态丢 CFG (少步蒸馏标配), 所以主表用 batch=1。

用法:
  CUDA_VISIBLE_DEVICES=5 python scripts/val/bench_decoder_phase0.py \
      --config configs/test_ar_model.yaml
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
from omegaconf import OmegaConf

XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path:
    sys.path.append(XNEMO_ROOT)

from diffusers import AutoencoderKLTemporalDecoder
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl


def build_decoder(config, device, dtype):
    """复刻 test_ar_model.load_xnemo_pipeline 的解码器构造 (去掉 AR/text/audio)。"""
    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(device=device, dtype=dtype).eval()
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=dtype, device=device).eval()
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config.vae_path).to(device=device, dtype=dtype).eval()

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(
        torch.load(config.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                   map_location="cpu"), strict=True)
    denoising_unet.load_state_dict(
        torch.load(config.temporal_module_path, map_location="cpu"), strict=False)
    return reference_unet, denoising_unet, vae


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def timeit(fn, warmup=3, iters=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))  # ms
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.mean()), float(ts.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    ap.add_argument("--frames", type=str, default="1,2,4,8,16,24")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"[env] torch={torch.__version__}  gpu={torch.cuda.get_device_name(0)}  dtype={args.dtype}")
    config = OmegaConf.load(args.config)

    print("[load] building decoder (reference + denoising 3D UNet + SVD VAE) ...")
    ref_unet, den_unet, vae = build_decoder(config, device, dtype)
    print(f"[params] reference_unet={n_params(ref_unet)/1e6:.1f}M  "
          f"denoising_unet={n_params(den_unet)/1e6:.1f}M  vae={n_params(vae)/1e6:.1f}M")

    H = W = 64  # 512px / 8
    F_list = [int(x) for x in args.frames.split(",")]

    # ---- reference attention 设置 (student 部署态: do_cfg=False) ----
    do_cfg = False
    writer = ReferenceAttentionControl(ref_unet, do_classifier_free_guidance=do_cfg,
                                       mode="write", batch_size=1, fusion_blocks="full")
    reader = ReferenceAttentionControl(den_unet, do_classifier_free_guidance=do_cfg,
                                       mode="read", batch_size=1, fusion_blocks="full")
    clip_emb = torch.randn(1, 1, 768, device=device, dtype=dtype)
    ref_latent = torch.randn(1, 4, H, W, device=device, dtype=dtype)
    t = torch.tensor(500, device=device)

    def run_reference():
        writer.clear()  # write 模式每次 forward 都 append bank, 计时前必须清空避免累积
        ref_unet(ref_latent, torch.zeros_like(t),
                 encoder_hidden_states=clip_emb, return_dict=False)

    ref_med, ref_mean, ref_std = timeit(run_reference, warmup=2, iters=8)
    # timeit 最后一次 run_reference 留下干净 1x bank → 复制给 reader 供逐 step 去噪复用
    reader.update(writer)
    print(f"\n[reference UNet 一次性] median={ref_med:.2f} ms (整片只跑一次, 摊销≈0)")

    # ---- 去噪 UNet 逐 F benchmark (batch=1, student 无 CFG) ----
    print(f"\n[denoising UNet forward]  H=W={H}  batch=1 (student, no CFG)")
    print(f"  {'F(frames)':>10} {'median(ms)':>12} {'mean':>8} {'std':>7} {'ms/frame':>9}")
    t_unet = {}
    reader.update(writer)  # bank 在整片去噪中静态, 只更新一次 (匹配真实推理), 不计入逐 step 计时
    for F in F_list:
        lat = torch.randn(1, 4, F, H, W, device=device, dtype=dtype)
        mot = torch.randn(1, F, 32, 16, device=device, dtype=dtype)

        def run_den():
            den_unet(lat, t, encoder_hidden_states=[clip_emb, mot],
                     pose_cond_fea=None, return_dict=False)
        med, mean, std = timeit(run_den, warmup=5, iters=15)
        t_unet[F] = med
        print(f"  {F:>10} {med:>12.2f} {mean:>8.2f} {std:>7.2f} {med/F:>9.2f}")

    # ---- CFG 开销对照 (batch=2) on F=4 ----
    try:
        writer2 = ReferenceAttentionControl(ref_unet, do_classifier_free_guidance=True,
                                            mode="write", batch_size=1, fusion_blocks="full")
        reader2 = ReferenceAttentionControl(den_unet, do_classifier_free_guidance=True,
                                            mode="read", batch_size=1, fusion_blocks="full")
        clip2 = torch.randn(2, 1, 768, device=device, dtype=dtype)
        ref2 = torch.randn(2, 4, H, W, device=device, dtype=dtype)
        writer2.clear()
        ref_unet(ref2, torch.zeros_like(t), encoder_hidden_states=clip2, return_dict=False)
        reader2.update(writer2)
        F = 4
        lat2 = torch.randn(2, 4, F, H, W, device=device, dtype=dtype)
        mot2 = torch.randn(2, F, 32, 16, device=device, dtype=dtype)

        def run_den_cfg():
            den_unet(lat2, t, encoder_hidden_states=[clip2, mot2],
                     pose_cond_fea=None, return_dict=False)
        med2, _, _ = timeit(run_den_cfg, warmup=5, iters=12)
        print(f"\n[CFG 对照] F=4 batch=2 (teacher CFG) median={med2:.2f} ms  "
              f"vs batch=1 {t_unet.get(4, float('nan')):.2f} ms  (≈{med2/max(t_unet.get(4,1),1e-9):.2f}x)")
        # 恢复 student reader
        reader.update(writer)
    except Exception as ex:
        print(f"[CFG 对照] 跳过 ({ex})")

    # ---- SVD VAE decode benchmark ----
    print(f"\n[SVD VAE decode]  64x64 latent -> 512x512")
    print(f"  {'F(frames)':>10} {'median(ms)':>12} {'ms/frame':>9}")
    t_vae = {}
    for F in F_list:
        z = torch.randn(F, 4, H, W, device=device, dtype=dtype) / 0.18215

        def run_vae():
            vae.decode(z, F)
        try:
            med, _, _ = timeit(run_vae, warmup=2, iters=8)
        except Exception as ex:
            med = float("nan")
            print(f"  [vae F={F}] err {ex}")
        t_vae[F] = med
        print(f"  {F:>10} {med:>12.2f} {med/F:>9.2f}")

    # ---- fps 表推导 ----
    print(f"\n{'='*72}")
    print("[fps 估算]  生成 K 帧耗时 = S*t_unet(K) + t_vae(K);  fps = K / 耗时")
    print("  (block=K 帧, S=去噪步数; UNet 无 CFG; reference 摊销忽略)")
    print(f"{'='*72}")
    K_list = [k for k in [1, 2, 4, 8] if k in t_unet]
    S_list = [1, 2, 4]
    header = "  " + "K\\S".rjust(6) + "".join([f"   S={s} (unet|+vae)".rjust(20) for s in S_list])
    print(header)
    for K in K_list:
        row = "  " + f"K={K}".rjust(6)
        for S in S_list:
            t_u = S * t_unet[K]
            t_full = t_u + (t_vae.get(K, 0.0) if not np.isnan(t_vae.get(K, np.nan)) else 0.0)
            fps_u = 1000.0 * K / t_u
            fps_full = 1000.0 * K / t_full if t_full > 0 else float("nan")
            row += f"{fps_u:>8.1f}|{fps_full:<10.1f}".rjust(20)
        print(row)
    print(f"\n  目标线: 25 fps. '+vae' 列含 SVD decode; 真实流式 decode 可与下一 block 的 UNet 流水并行。")
    print(f"  注: 此为 block 内无历史上下文的下界估计; 因果流式稳态会因 temporal-attn 读 KV-cache 略增 "
          f"(temporal module 占比小, 影响有限)。peak mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()

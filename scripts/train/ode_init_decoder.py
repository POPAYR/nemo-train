"""
Phase 2: ODE 初始化 (DECODER_DISTILL_PLAN.md §Phase 2)
======================================================
把因果 student（block-causal）从 teacher 权重稳定引导出来（DMD2 前置）：
eps-MSE 回归 + block-causal(block=8)，把双向 temporal 模块改成因果、修 cold-start，保持多步能力。
**降步到 1-step 交给 Phase 3 DMD2**（回归到 1-step x0 = DMD 要修的均值模糊）。

数据：**直接复用 train_ar_xnemo.py 的加载方式**——`MotarDataset(load_video=True)` + `ConcatDataset`，
源(MEAD+hallo3)与参数读 `configs/train_ar.yaml` 的 data 段（含 data_name_path 索引，不用扫盘）。
batch 关键字：motion_tensor(归一化→denorm 喂 UNet) / video_tensor(=x0) / ref_latent / ref_img(→CLIP) / mask。
只训 `temporal_modules`，spatial/motion/reference/CLIP 全冻结。

用法:  CUDA_VISIBLE_DEVICES=5 python scripts/train/ode_init_decoder.py --smoke
       CUDA_VISIBLE_DEVICES=5 python scripts/train/ode_init_decoder.py --max_steps 40000 --batch 6 &
"""
import os, sys, argparse, glob, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")          # AR_REPO_ROOT，供 MotarDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers import DDIMScheduler
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"   # data 段(源/索引/stats)
DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"  # 解码器权重路径


def build(dev, dt):
    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
    denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
    sched = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
    return refu, denu, imgenc, sched


def build_loader(args, dev, dt):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)

    def make(src):
        return MotarDataset(
            pose_dir=src.pose_dir, audio_dir=src.audio_dir, caption_dir=src.caption_dir,
            data_name_path=src.data_name_path, tokenizer_path=dcfg.tokenizer_path,
            data_stats_path=dcfg.data_stats_path,
            context_length=args.L, fps=dcfg.fps, sr=dcfg.sr,
            text_max_len=dcfg.get("text_max_len", 128),
            random_crop=True, pad_short=True, load_video=True,
            latent_dir=src.latent_dir, video_dir=src.video_dir, video_processor=vproc)

    ds = [make(s) for s in dcfg.sources]
    train_ds = ConcatDataset(ds) if len(ds) > 1 else ds[0]
    print(f"[data] {len(dcfg.sources)} sources(MEAD+hallo3) concat → {len(train_ds)} clips")
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True, prefetch_factor=3, persistent_workers=True)
    stats = torch.load(dcfg.data_stats_path, map_location="cpu")
    mean = stats["mean"].reshape(-1).to(dev, dt); std = stats["std"].reshape(-1).to(dev, dt)
    return dl, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-5)
    # ★ EMA:见 src/utils/ema.py —— 无 EMA 时权重在小 batch 下游走,单点指标不可信
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6, help="micro-batch(clip 数); A100 显存够可调大")
    ap.add_argument("--train_scope", choices=["temporal", "all"], default="temporal")
    ap.add_argument("--t_low", type=int, default=20)
    ap.add_argument("--t_high", type=int, default=980)
    ap.add_argument("--out", default="/media/ps/ssd5/ayr/x-nemo-inference/output/ode_init")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_last", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.batch = 30, 10000, 5, 1, 2

    dev = torch.device("cuda:0"); dt = torch.bfloat16
    os.makedirs(args.out, exist_ok=True)
    refu, denu, imgenc, sched = build(dev, dt)
    acp = sched.alphas_cumprod.to(dev)

    denu.requires_grad_(False)
    if args.train_scope == "all":
        denu.requires_grad_(True)
    else:  # 只训 temporal_modules（因果适配核心；spatial/motion/reference 冻结，保留 teacher 外观/口型）
        for nm, p in denu.named_parameters():
            if "temporal_modules" in nm:
                p.requires_grad_(True)
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    train_params = [p for p in denu.parameters() if p.requires_grad]
    denu.train()
    print(f"[scope] {args.train_scope}  trainable={sum(p.numel() for p in train_params)/1e6:.1f}M")

    ctrl = TemporalCausalControl(denu, block_size=args.block, window=0)
    ctrl.set_mode("train")
    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=args.batch, fusion_blocks="full")

    dl, m_mean, m_std = build_loader(args, dev, dt)
    def denorm(x): return x * (m_std + 1e-6) + m_mean
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    from src.utils.ema import EMA, ema_weights
    ema = EMA(denu, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    print(f"[ema] {'on decay=%.4f' % args.ema_decay if ema else 'off'}", flush=True)

    step = 0; t_last = time.time(); loss_acc = 0.0; checked = False
    data_iter = iter(dl)
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for _ in range(args.accum):
            try: b = next(data_iter)
            except StopIteration:
                data_iter = iter(dl); b = next(data_iter)
            x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W] (=UNet 空间 latent)
            B, T = x0.shape[0], x0.shape[2]
            motion = denorm(b["motion_tensor"].to(dev, dt)).reshape(B, T, 32, 16)   # 归一化→raw 喂 UNet
            ref_latent = b["ref_latent"].to(dev, dt)
            mask = b.get("mask", None)
            if mask is not None:
                mask = mask.to(dev, dt).view(B, 1, T, 1, 1)
            if not checked:  # 一次性核对 latent 缩放（应 ~1 = UNet 空间）
                print(f"[check] video_tensor.std={x0.float().std():.3f} ref_latent.std={ref_latent.float().std():.3f} "
                      f"motion(raw).std={motion.float().std():.3f}")
                checked = True

            with torch.no_grad():
                clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
                rwriter.clear()
                refu(ref_latent, torch.zeros((), device=dev).long(), encoder_hidden_states=clip_emb, return_dict=False)
                rreader.update(rwriter, dtype=dt)

            t = torch.randint(args.t_low, args.t_high, (1,), device=dev).long()
            noise = torch.randn_like(x0)
            x_t = sched.add_noise(x0, noise, t).to(dt)
            eps = denu(x_t, t, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
            se = (eps.float() - noise.float()) ** 2
            loss = (se * mask).sum() / mask.expand_as(se).sum().clamp_min(1) if mask is not None else se.mean()
            loss = loss / args.accum
            loss.backward(); micro += loss.item()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(denu)
        step += 1; loss_acc += micro
        if step % args.log_every == 0:
            dt_s = (time.time() - t_last) / args.log_every; t_last = time.time()
            print(f"step {step:6d}  eps_mse={loss_acc/args.log_every:.5f}  {dt_s:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            loss_acc = 0.0
        if step % args.save_every == 0 or step == args.max_steps:
            torch.save({**({"ema": ema.state_dict()} if ema is not None else {}), "denoising_unet": denu.state_dict(), "step": step, "args": vars(args)},
                       f"{args.out}/ode_step_{step}.pt")
            print(f"[save] {args.out}/ode_step_{step}.pt", flush=True)
            cks = sorted(glob.glob(f"{args.out}/ode_step_*.pt"),
                         key=lambda p: int(os.path.basename(p)[len('ode_step_'):-3]))
            for old in cks[:-args.keep_last]:
                try: os.remove(old); print(f"[rmckpt] {old}", flush=True)
                except OSError: pass
    print("[done]")


if __name__ == "__main__":
    main()

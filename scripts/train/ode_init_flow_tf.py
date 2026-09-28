"""
Phase 2.5-TF: 因果 flow 初始化 —— teacher forcing 流式版
=========================================================
与 ode_init_flow.py(并行 block-causal + 整段单一 σ)的区别:**历史用 GT 干净帧**。
理由:Self-Forcing 推理时 KV cache 里的历史恒为干净 x0(官方 context_noise=0),
      而并行单-σ 训练里历史与当前同噪声 —— 这个配置推理时一次都不出现。
      本脚本逐 block 训练,历史 = 已 commit 的 GT 干净块,当前 block 带噪 → 与推理输入精确一致
      (唯一残差:历史是 GT 而非模型自己的预测,那一步交给后续 SF-DMD)。

每个 micro-batch:
    ctrl.set_mode("stream"); reset_cache()
    for k in blocks:
        σ_k ~ logit-normal                       # 每块独立(历史始终干净,非 diffusion forcing)
        z_k = (1-σ_k)·x0_k + σ_k·ε
        v_pred = unet(z_k, σ_k·1000)             # 带梯度;历史来自 cache(干净 GT)
        loss_k = ‖v_pred − (ε − x0_k)‖²  → 立即 backward(块间无梯度耦合,省显存)
        commit: unet(x0_gt_k, σ=0) under no_grad # 把 GT 干净块写进 KV
只训 temporal_modules(spatial 冻结,保住 flow teacher 的锐利外观)。

用法: CUDA_VISIBLE_DEVICES=4 python scripts/train/ode_init_flow_tf.py --smoke
      CUDA_VISIBLE_DEVICES=4 python scripts/train/ode_init_flow_tf.py --max_steps 4000 --L 64 --batch 1 --accum 8
"""
import os, sys, argparse, glob, time
import torch
import torch.distributed as dist
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
FLOW_TEACHER = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_teacher/flow_teacher_FINAL.pt"

import torch.utils.checkpoint as _ckptmod
_orig_ckpt = _ckptmod.checkpoint
def _ckpt_nonreentrant(fn, *a, use_reentrant=None, **k):
    return _orig_ckpt(fn, *a, use_reentrant=False, **k)


def build(dev, dt, flow_ckpt):
    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
    denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    raw = torch.load(flow_ckpt, map_location="cpu")
    fsd = raw["denoising_unet"] if "denoising_unet" in raw else raw
    denu.load_state_dict(fsd, strict=False)
    print(f"[flow] denu ← {flow_ckpt} (step={raw.get('step')} obj={raw.get('objective')})", flush=True)
    imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
    return refu, denu, imgenc


def build_loader(args, dev, dt, rank=0, world=1):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    def make(src):
        return MotarDataset(pose_dir=src.pose_dir, audio_dir=src.audio_dir, caption_dir=src.caption_dir,
            data_name_path=src.data_name_path, tokenizer_path=dcfg.tokenizer_path,
            data_stats_path=dcfg.data_stats_path, context_length=args.L, fps=dcfg.fps, sr=dcfg.sr,
            text_max_len=dcfg.get("text_max_len", 128), random_crop=True, pad_short=True, load_video=True,
            latent_dir=src.latent_dir, video_dir=src.video_dir, video_processor=vproc)
    ds = ConcatDataset([make(s) for s in dcfg.sources])
    sampler = None
    if world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None), num_workers=8,
                    pin_memory=True, drop_last=True, prefetch_factor=3, persistent_workers=True)
    stats = torch.load(dcfg.data_stats_path, map_location="cpu")
    return dl, stats["mean"].reshape(-1).to(dev, dt), stats["std"].reshape(-1).to(dev, dt)


def sample_sigma(B, dev, mode="uniform"):
    # ★ 2026-09-13 起**写死 uniform**,lognorm 分支已删除。
    # 原因见 docs/decisions/2026-09-12_sigma-distribution.md:
    # lognorm(=sigmoid(randn)) 配 shift=3 时 σ<0.25 只占 **1.4%** 训练样本,
    # 细节精修段严重欠训 → 帧间跳变;而 uniform 配同样的 shift=3 得到
    # σ<.25/.25-.5/.5-.75/.75-.9/.9-1 = 10.0/15.0/25.0/25.0/25.1%,
    # **σ≥0.75 仍是 50%**(4 步学生的需求不受损),是纯增益。
    # 曾经的注释把 uniform 的占比写成了 lognorm 的,导致这个问题长期未被发现。
    assert mode == "uniform", f"σ 分布已写死 uniform,收到 {mode!r}"
    return torch.rand(B, device=dev).clamp(1e-3, 1 - 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    # ★ EMA:见 src/utils/ema.py —— 无 EMA 时权重在小 batch 下游走,单点指标不可信
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--train_scope", choices=["temporal", "temporal_cross", "all"], default="temporal")
    ap.add_argument("--t_mode", choices=["uniform"], default="uniform",
                    help="σ 分布已写死 uniform;传 lognorm 会被 argparse 直接拒绝")
    ap.add_argument("--flow_ckpt", default=FLOW_TEACHER)
    ap.add_argument("--out", default="/media/ps/ssd5/ayr/x-nemo-inference/output/ode_init_flow_tf")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_last", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.batch, args.L = 10, 10000, 2, 1, 1, 32

    _ckptmod.checkpoint = _ckpt_nonreentrant

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        lrk = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrk); dev = torch.device(f"cuda:{lrk}")
    else:
        rank, world, dev = 0, 1, torch.device("cuda:0")
    is_main = (rank == 0); dt = torch.bfloat16
    if is_main: os.makedirs(args.out, exist_ok=True)

    refu, denu, imgenc = build(dev, dt, args.flow_ckpt)

    denu.requires_grad_(False)
    if args.train_scope == "all":
        denu.requires_grad_(True)
    elif args.train_scope == "temporal_cross":
        for nm, p in denu.named_parameters():
            if "temporal_modules" in nm or "attn2" in nm: p.requires_grad_(True)
    else:
        for nm, p in denu.named_parameters():
            if "temporal_modules" in nm: p.requires_grad_(True)
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    train_params = [p for p in denu.parameters() if p.requires_grad]
    denu.train()
    NB = args.L // args.block
    if is_main:
        print(f"[scope] {args.train_scope}  trainable={sum(p.numel() for p in train_params)/1e6:.1f}M  "
              f"world={world} batch={args.batch} accum={args.accum} → 有效bsz={args.batch*args.accum*world}", flush=True)
        print(f"[tf] teacher-forcing 流式: L={args.L} block={args.block} → {NB} blocks/clip, 历史=GT干净帧", flush=True)

    ctrl = TemporalCausalControl(denu, block_size=args.block, window=0)
    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=args.batch, fusion_blocks="full")

    dl, m_mean, m_std = build_loader(args, dev, dt, rank, world)
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
            except StopIteration: data_iter = iter(dl); b = next(data_iter)
            x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()   # [B,C,T,H,W]
            B, T = x0.shape[0], x0.shape[2]
            motion = denorm(b["motion_tensor"].to(dev, dt)).reshape(B, T, 32, 16)
            ref_latent = b["ref_latent"].to(dev, dt)
            mask = b.get("mask", None)
            if mask is not None: mask = mask.to(dev, dt).view(B, 1, T, 1, 1)

            with torch.no_grad():
                clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
                rwriter.clear()
                refu(ref_latent, torch.zeros((), device=dev).long(), encoder_hidden_states=clip_emb, return_dict=False)
                rreader.update(rwriter, dtype=dt)

            # ---- teacher-forcing 流式逐 block ----
            ctrl.set_mode("stream"); ctrl.reset_cache()
            nblk = T // args.block
            for bi in range(nblk):
                sl = slice(bi * args.block, (bi + 1) * args.block)
                x0_b = x0[:, :, sl]; mot_b = motion[:, sl]
                sig = sample_sigma(B, dev, args.t_mode)                     # 每块独立 σ(历史恒干净)
                sb = sig.view(B, 1, 1, 1, 1)
                noise = torch.randn_like(x0_b)
                z_b = ((1 - sb) * x0_b.float() + sb * noise.float()).to(dt)
                v_tgt = (noise.float() - x0_b.float())
                t_emb = (sig * 1000.0).to(dt)

                ctrl.set_offset(bi * args.block)
                ctrl.set_commit(False)                                      # 只读 cache(干净 GT 历史)
                v_pred = denu(z_b, t_emb, encoder_hidden_states=[clip_emb, mot_b],
                              pose_cond_fea=None, return_dict=False)[0]
                se = (v_pred.float() - v_tgt) ** 2
                if mask is not None:
                    mk = mask[:, :, sl]
                    lb = (se * mk).sum() / mk.expand_as(se).sum().clamp_min(1)
                else:
                    lb = se.mean()
                # 块间无梯度耦合(历史 cache 在 no_grad 下写入)→ 逐块 backward,显存只占 1 块
                (lb / (args.accum * nblk)).backward()
                micro += lb.item() / (args.accum * nblk)

                if not checked and is_main and bi == 0:
                    print(f"[check] x0.std={x0.float().std():.3f} v_target.std={v_tgt.std():.3f} "
                          f"v_pred.std={v_pred.float().std():.3f} sigma=[{sig.min():.3f},{sig.max():.3f}] "
                          f"nblk={nblk}", flush=True); checked = True

                # commit:GT 干净块写进 KV cache(σ=0)
                ctrl.set_commit(True)
                with torch.no_grad():
                    t0 = torch.zeros(B, device=dev, dtype=dt)
                    denu(x0_b, t0, encoder_hidden_states=[clip_emb, mot_b],
                         pose_cond_fea=None, return_dict=False)
            ctrl.set_mode("off"); ctrl.reset_cache()

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(denu)
        step += 1; loss_acc += micro
        if step % args.log_every == 0 and is_main:
            dt_s = (time.time() - t_last) / args.log_every; t_last = time.time()
            print(f"step {step:6d}  tf_vmse={loss_acc/args.log_every:.5f}  {dt_s:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            loss_acc = 0.0
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            torch.save({**({"ema": ema.state_dict()} if ema is not None else {}), "denoising_unet": denu.state_dict(), "step": step, "args": vars(args),
                        "objective": "rectified_flow_v_causal_tf"}, f"{args.out}/flow_tf_step_{step}.pt")
            print(f"[save] {args.out}/flow_tf_step_{step}.pt", flush=True)
            cks = sorted(glob.glob(f"{args.out}/flow_tf_step_*.pt"),
                         key=lambda p: int(os.path.basename(p)[len('flow_tf_step_'):-3]))
            for old in cks[:-args.keep_last]:
                try: os.remove(old); print(f"[rmckpt] {old}", flush=True)
                except OSError: pass
    if is_main: print("[done]")


if __name__ == "__main__":
    main()

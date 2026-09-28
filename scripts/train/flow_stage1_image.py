"""
Stage 1:image backbone 的 rectified-flow 适配(**完全移除 temporal module**)
==========================================================================
动机(实测,见 scripts/val 的 diag):此前用 train_scope=all 在 64 帧视频目标上做 ε→flow 微调,
优化器把工作卸载给了 temporal 模块,导致空间主干**失去独立工作能力**:
    「仅空间 / 完整模型」的 loss 比值   原始 ε = 1.29x    我们的 flow = 3.54x
空间不独立 → temporal 承担了它不该承担的职责 → 一动 temporal(换 PE)就全塌。

本阶段复刻 XNeMo 原始 stage 1 的训练契约:
  - `use_temporal_module=False` —— temporal self-attn **不构建、不前向、不接梯度**;
    保留 `use_motion_module`(Spatial_Cross,cross_attention_dim=16,即逐帧 motion 条件)。
  - 于是每帧完全独立 → 纯粹的「参考图 + motion token → 该帧」图生图任务。
  - **cross-reconstruction:参考帧与目标帧都从整段视频随机抽**(src/data/image_pair_dataset.py),
    不受滑动窗口限制 —— 姿态/表情差可跨整段视频,且同 batch 内样本互不相关,泛化性远好于窗口法。
  - motion 用预计算 pose latent(motion encoder 天然冻结),只训图像模型,因此很快。
  - 全量微调 image backbone,目标 rectified flow:z=(1-σ)x0+σε,v=ε-x0,t_emb=σ·1000。
起点默认用**原始 ε 空间权重**(而非已被协同适配污染的 flow 权重)。

监控:除 flow_mse 外,本脚本天然就是「仅空间」的 loss —— 它就是 stage2 要守住的健康度指标。

用法:
  CUDA_VISIBLE_DEVICES=2 python scripts/train/flow_stage1_image.py --smoke
  CUDA_VISIBLE_DEVICES=2,5 torchrun --nproc_per_node=2 --master_port 29590 \
      scripts/train/flow_stage1_image.py --L 16 --batch 4 --accum 4 --lr 1e-5
"""
import os, sys, argparse, glob, time
import torch
import torch.distributed as dist
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
sys.path.append(third_party("motar"))
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from src.data.image_pair_dataset import ImagePairDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl

TRAIN_CFG = XP("REPO", "configs/train_ar.yaml")
DEC_CFG = XP("REPO", "configs/test_ar_model.yaml")

import torch.utils.checkpoint as _ckptmod
_orig_ckpt = _ckptmod.checkpoint
def _ckpt_nonreentrant(fn, *a, use_reentrant=None, **k):
    return _orig_ckpt(fn, *a, use_reentrant=False, **k)


def build(dev, dt, from_ckpt=None):
    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    uak = OmegaConf.to_container(ic.unet_additional_kwargs, resolve=True)
    uak["use_temporal_module"] = False          # ★ 彻底移除 temporal self-attn(motion cross-attn 保留)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                    map_location="cpu"), strict=True)
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=uak).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    if from_ckpt:
        sd = torch.load(from_ckpt, map_location="cpu")
        sd = sd.get("denoising_unet", sd)
        sd = {k: v for k, v in sd.items() if "temporal_modules" not in k}
        denu.load_state_dict(sd, strict=False)
    imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
    n_tmp = sum(1 for n, _ in denu.named_parameters() if "temporal_modules" in n)
    return refu, denu, imgenc, n_tmp


def build_loader(args, dev, dt, rank=0, world=1):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    # ★ §8.0:训练集只用 hallo3(--sources 0);MEAD 仅作 eval 参考
    _srcs = list(dcfg.sources) if args.sources is None else [dcfg.sources[i] for i in args.sources]
    ds = ConcatDataset([ImagePairDataset(
        # ★ --pose_real:改用真实逐帧 bbox 的 motion latent;--bbox_drop:按概率混入常量版
        pose_dir=(XP("XN_HALLO3", "pose_embed_real") if args.pose_real else s.pose_dir),
        pose_dir_alt=(XP("XN_HALLO3", "pose_embed") if args.pose_real else None), bbox_drop=args.bbox_drop,
        latent_dir=s.latent_dir, video_dir=s.video_dir,
        data_name_path=s.data_name_path, n_target=args.n_target,
        min_gap=args.min_gap, video_processor=vproc) for s in _srcs])
    sampler = None
    if world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    prefetch_factor=4, persistent_workers=True)
    return dl


def sample_sigma(B, dev, mode="uniform", shift=1.0):
    """训练时的噪声档采样。

    ★ shift 的作用(SD3/Flux, Wan2.1/2.2 训练时用 shift=5):
      σ ← s·σ/(1+(s-1)σ),把采样质量挤向高噪段。
      理由:维度越高,同一个 σ 破坏的信息越少,故高维数据需要在更高的 σ 上多训。
      我们原来是 shift=1(logit-normal 中位数 0.499),σ>0.75 只占 12.2%、σ>0.9 只占 1.4%,
      而学生 4 步推理有 50%~75% 的步落在 σ≥0.75 —— 训练/推理严重错配。
      这也是"官方 score_shift=5.0 照搬会退化、打分 σ 须截断≤0.70"的根因:
      teacher 在高噪段没训够,在那里打分等于拿噪声当监督。
      shift=3 时各段占比 ≈ 9.9/15.1/25.0/24.9/25.1%(区间 .25/.5/.75/.9/1.0)。
    """
    # ★ 2026-09-13 起**写死 uniform**,lognorm 分支已删除。
    # 原因见 docs/decisions/2026-09-12_sigma-distribution.md:
    # lognorm(=sigmoid(randn)) 配 shift=3 时 σ<0.25 只占 **1.4%** 训练样本,
    # 细节精修段严重欠训 → 帧间跳变;而 uniform 配同样的 shift=3 得到
    # σ<.25/.25-.5/.5-.75/.75-.9/.9-1 = 10.0/15.0/25.0/25.0/25.1%,
    # **σ≥0.75 仍是 50%**(4 步学生的需求不受损),是纯增益。
    # 曾经的注释把 uniform 的占比写成了 lognorm 的,导致这个问题长期未被发现。
    assert mode == "uniform", f"σ 分布已写死 uniform,收到 {mode!r}"
    s = torch.rand(B, device=dev)
    if shift != 1.0:
        s = shift * s / (1.0 + (shift - 1.0) * s)
    return s.clamp(1e-3, 1 - 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-5)
    # ★ EMA:见 src/utils/ema.py —— 无 EMA 时权重在小 batch 下游走,单点指标不可信
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--n_target", type=int, default=4,
                    help="每个参考帧配几个随机目标帧。帧从整段视频随机抽,彼此不相关;"
                         "N 越小样本越多样,但 reference-UNet 的开销摊薄越少(N=4 时约 20% 开销)")
    ap.add_argument("--pose_real", action="store_true",
                    help="用真实逐帧 bbox 的 motion latent(pose_embed_real);\n                          背景运动本无条件信号,导致背景乱动,见 §5d")
    ap.add_argument("--bbox_drop", type=float, default=0.0,
                    help="按此概率改用常量 [0,0,1] 版,让推理时的输入也在训练分布内")
    ap.add_argument("--sources", type=int, nargs="+", default=None,
                    help="用哪些数据源(0=hallo3 1=MEAD)。§8.0 目标域为 hallo3,应传 0")
    ap.add_argument("--min_gap", type=int, default=0, help="目标帧与参考帧的最小间隔(0=不限)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--t_mode", choices=["uniform"], default="uniform",
                    help="σ 分布已写死 uniform;传 lognorm 会被 argparse 直接拒绝(而非静默退回旧行为)")
    ap.add_argument("--cfg_drop", type=float, default=0.0,
                    help="CFG condition dropout 概率。★ stage1 原本没有,是设计缺陷:"
                         "XNeMo 三路置空(bank不注入+CLIP置零+参考帧motion)**全部作用在空间主干**上,"
                         "而 stage2 冻结主干 —— stage1 是唯一能让主干在 flow 参数化下学会 uncond 形态的地方。"
                         "只在 stage2 开 cfg_drop 等于让 temporal 去补主干的短板。建议与 stage2 取同值 0.1")
    ap.add_argument("--sigma_shift", type=float, default=1.0,
                    help="训练 σ 采样的 shift(SD3/Wan 口径)。1.0=原状。★ 必须与下游"
                         "gen_ode_pairs --shift / distill --flow_grid_shift 保持一致")
    ap.add_argument("--val_only", default=None,
                    help="glob 模式:对匹配的每个 ckpt 跑完整验证后退出(选最优点用)")
    ap.add_argument("--from_ckpt", default=None, help="缺省=原始 ε 空间权重;也可从已有 flow 空间权重续训")
    ap.add_argument("--out", default=XP("XN_OUTPUT", "flow_stage1"))
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--keep_last", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--val_every", type=int, default=0,
                    help="每 N 步在固定验证子集上算 v-MSE(便宜)。0=关闭。见 CLAUDE.md §6.4")
    ap.add_argument("--val_sample_every", type=int, default=0,
                    help="每 N 步额外做采样+VAE解码,算 PSNR/SSIM/LPIPS/FID 并存可视化。0=从不采样")
    ap.add_argument("--val_clips", type=int, default=8)
    # ★ 评测必须在交付工作点上做(原来固定 cfg=1.0 导致跨线排序判反,见 DATA.md §22)
    ap.add_argument("--val_cfg", type=float, default=2.0,
                    help="验证采样用的 CFG。1.0=无引导(旧行为,与历史数字可比)")
    ap.add_argument("--val_sample_steps", type=int, default=20, help="验证采样步数(求快,不必等于部署步数)")
    ap.add_argument("--val_frames", type=int, default=4)

    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.batch = 20, 10000, 5, 1, 2

    _ckptmod.checkpoint = _ckpt_nonreentrant
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        # ★ 超时放到 4h:验证只在 rank0 跑,其余 rank 在 barrier 空等。
        #   默认 30min 在机器高负载时不够(step0 的 8 clip 采样就超了),
        #   会被 NCCL 判成挂死、发 SIGABRT 直接崩掉整个训练。
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        rank = dist.get_rank(); world = dist.get_world_size()
        lrk = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrk); dev = torch.device(f"cuda:{lrk}")
    else:
        rank, world, dev = 0, 1, torch.device("cuda:0")
    is_main = (rank == 0); dt = torch.bfloat16
    if is_main: os.makedirs(args.out, exist_ok=True)

    refu, denu, imgenc, n_tmp = build(dev, dt, args.from_ckpt)
    assert n_tmp == 0, f"temporal_modules 未被移除,仍有 {n_tmp} 个参数"
    denu.requires_grad_(True)                    # 全量微调 image backbone
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    train_params = [p for p in denu.parameters() if p.requires_grad]
    denu.train()
    if is_main:
        print(f"[stage1] temporal_module 已彻底移除(temporal 参数数={n_tmp})", flush=True)
        print(f"[scope] 全量 image backbone  trainable={sum(p.numel() for p in train_params)/1e6:.1f}M  "
              f"world={world} batch={args.batch} accum={args.accum} → 每步 {args.batch*args.accum*world} 个参考帧 "
              f"× {args.n_target} 随机目标帧 = {args.batch*args.accum*world*args.n_target} 帧"
              f"(参考/目标均从整段视频随机抽)", flush=True)
        print(f"[init] {'从 '+args.from_ckpt if args.from_ckpt else '原始 ε 空间权重(xnemo_denoising_unet)'}", flush=True)
        print(f"[cfg_drop] {args.cfg_drop}  (>0 时按此概率训 uncond:bank不注入+CLIP置零+参考帧motion)", flush=True)

    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=args.batch, fusion_blocks="full")
    if distributed:
        denu = torch.nn.parallel.DistributedDataParallel(denu, device_ids=[lrk], output_device=lrk,
                broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
        denu_core = denu.module
    else:
        denu_core = denu

    dl = build_loader(args, dev, dt, rank, world)
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    from src.utils.ema import EMA, ema_weights
    ema = EMA(denu_core, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    print(f"[ema] {'on decay=%.4f' % args.ema_decay if ema else 'off'}", flush=True)


    # ---- 训练内验证(CLAUDE.md §6.4) + TensorBoard
    _tb = None; _val = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        _tb = SummaryWriter(os.path.join(args.out, "logs", "tb"))
    if args.val_every > 0:
        from src.utils.inproc_val import Validator
        _val = Validator(out_dir=args.out, kind="image", n_clips=args.val_clips,
                         n_frames=args.val_frames,
                         sample_steps=args.val_sample_steps,
                         sample_shift=args.sigma_shift, val_cfg=args.val_cfg, is_main=is_main, tb=_tb)

    # ★ --val_only:对一批 ckpt 逐个跑**完整**验证,用于选最优点而不是无脑取最后一个。
    #   走训练脚本自身的 build/Validator,保证评测前向与训练完全一致
    #   (评测脚本另写一套曾导致口径不一致,见 FLOW_DISTILL_PROGRESS §5 教训 19)。
    if args.val_only:
        import glob as _g, re as _re
        cks = sorted(_g.glob(args.val_only),
                     key=lambda x: int(_re.search(r"step_(\d+)", x).group(1)))
        for ck in cks:
            st = int(_re.search(r"step_(\d+)", ck).group(1))
            sd = torch.load(ck, map_location="cpu"); sd = sd.get("denoising_unet", sd)
            denu_core.load_state_dict(sd, strict=False)
            with ema_weights(ema, denu_core):
                _val.run(st, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter,
                         do_sample=True)
        if is_main: print("[done] val_only", flush=True)
        return

    step = 0; t_last = time.time(); loss_acc = 0.0; checked = False
    it = iter(dl)
    # ★ 训练前先跑一次 step 0 基线,否则看不到第一段的跃升
    if args.val_every > 0 and is_main:
        with ema_weights(ema, denu_core):
            _val.run(0, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter,
                     do_sample=args.val_sample_every > 0)

    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for _ in range(args.accum):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x0 = b["tgt_latent"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()   # [B,C,N,H,W]
            B, T = x0.shape[0], x0.shape[2]
            motion = b["tgt_motion"].to(dev, dt)                                     # [B,N,32,16] 原始值
            mask = None
            # ★ CFG condition dropout:与 flow_stage2_temporal.py:194-204 逐条一致
            drop = (args.cfg_drop > 0) and (torch.rand(1).item() < args.cfg_drop)
            with torch.no_grad():
                clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
                rwriter.clear()
                refu(b["ref_latent"].to(dev, dt), torch.zeros((), device=dev).long(),
                     encoder_hidden_states=clip_emb, return_dict=False)
                rreader.update(rwriter, dtype=dt)
                if drop:
                    rreader.clear()                                          # ① bank 不注入
                    clip_emb = torch.zeros_like(clip_emb)                    # ② CLIP 置零
                    motion = motion[:, 0:1].expand_as(motion).contiguous()   # ③ 参考帧 motion

            sig = sample_sigma(B, dev, args.t_mode, args.sigma_shift).view(B, 1, 1, 1, 1)
            t_emb = (sig.view(B) * 1000.0).to(dt)
            noise = torch.randn_like(x0)
            z = ((1 - sig) * x0.float() + sig * noise.float()).to(dt)
            v_tgt = (noise.float() - x0.float())
            if not checked and is_main:
                print(f"[check] x0.std={x0.float().std():.3f} v_target.std={v_tgt.std():.3f} "
                      f"目标帧数={T}  参考-目标间隔 mean={b['gap'].float().mean():.1f} "
                      f"max={b['gap'].float().max():.0f} 帧", flush=True); checked = True

            v_pred = denu(z, t_emb, encoder_hidden_states=[clip_emb, motion],
                          pose_cond_fea=None, return_dict=False)[0]
            se = (v_pred.float() - v_tgt) ** 2
            loss = (se * mask).sum() / mask.expand_as(se).sum().clamp_min(1) if mask is not None else se.mean()
            (loss / args.accum).backward(); micro += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(denu_core)
        step += 1; loss_acc += micro
        if step % args.log_every == 0 and is_main:
            dts = (time.time() - t_last) / args.log_every; t_last = time.time()
            print(f"step {step:6d}  spatial_flow_mse={loss_acc/args.log_every:.5f}  {dts:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            if _tb is not None:
                _tb.add_scalar("train/loss", loss_acc / args.log_every, step)
            loss_acc = 0.0
        _do_s = args.val_sample_every > 0 and step % args.val_sample_every == 0
        # ★ 采样档独立判定:val_sample_every 不必是 val_every 的倍数
        #   (曾因嵌套判定导致 --val_sample_every 250 只在 500/1000/... 触发)
        if args.val_every > 0 and (step % args.val_every == 0 or _do_s):
            if is_main:
                with ema_weights(ema, denu_core):
                    _val.run(step, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter,
                             do_sample=_do_s)
            if distributed:
                dist.barrier()          # 其余 rank 等主进程验证完;验证集很小,开销可忽略
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            torch.save({**({"ema": ema.state_dict()} if ema is not None else {}), "denoising_unet": denu_core.state_dict(), "step": step, "args": vars(args),
                        "objective": "rectified_flow_v", "stage": 1, "no_temporal": True},
                       f"{args.out}/stage1_step_{step}.pt")
            print(f"[save] {args.out}/stage1_step_{step}.pt", flush=True)
            cks = sorted(glob.glob(f"{args.out}/stage1_step_*.pt"),
                         key=lambda p: int(os.path.basename(p)[len('stage1_step_'):-3]))
            for old in cks[:-args.keep_last]:
                try: os.remove(old); print(f"[rmckpt] {old}", flush=True)
                except OSError: pass
    if is_main: print("[done]")


if __name__ == "__main__":
    main()

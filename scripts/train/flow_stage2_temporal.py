"""
Stage 2:插入 RoPE 版 temporal module,**冻结 image backbone**,从零训练帧间平滑
================================================================================
承接 Stage 1(scripts/train/flow_stage1_image.py):image backbone 已独立完成 flow 适配
(仅空间 flow_mse ≈0.29,4 步可用)。本阶段复刻 XNeMo 原始 stage 2 的训练契约:

  - image backbone(spatial + motion cross-attn)**全部冻结** —— 防止重蹈协同适配陷阱:
    上一轮 train_scope=all 在视频目标上放开全网,优化器把活卸载给 temporal,
    空间主干失去独立能力(仅空间/完整 loss 比值 1.29x→3.54x)。冻结即从结构上杜绝。
  - temporal module **从零开始**(不加载 xnemo_temporal_module.pth)。
    `temporal_transformer.proj_out` 是 zero_module 初始化 → 起点即恒等 = 纯逐帧模型,
    无需「先忘掉加性 PE」的包袱(这正是 RoPE 微调卡在 +18.5% 的原因)。
  - 位置编码用 **RoPE**(`--rope`):相对位置,train==stream 数值一致、任意长度可外推。
    验证见 scripts/val/test_rope_parity.py(相对性/对拍/外推 三项)。
  - 只有 temporal 可训(453M),目标仍是 rectified flow v-MSE,但需要**长序列**才能学到平滑。

用法:
  CUDA_VISIBLE_DEVICES=2 python scripts/train/flow_stage2_temporal.py --smoke
  CUDA_VISIBLE_DEVICES=2,5 torchrun --nproc_per_node=2 --master_port 29595 \
      scripts/train/flow_stage2_temporal.py --stage1_ckpt output/flow_stage1/stage1_step_3000.pt \
      --rope --L 64 --batch 2 --accum 4 --lr 1e-4
"""
import os, sys, argparse, glob, time, math
import numpy as np
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
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope

TRAIN_CFG = XP("REPO", "configs/train_ar.yaml")
DEC_CFG = XP("REPO", "configs/test_ar_model.yaml")

import torch.utils.checkpoint as _ckptmod
_orig_ckpt = _ckptmod.checkpoint
def _ckpt_nonreentrant(fn, *a, use_reentrant=None, **k):
    return _orig_ckpt(fn, *a, use_reentrant=False, **k)


def build(dev, dt, stage1_ckpt, use_rope, resume=None, resume_temporal_only=False, use_ema_weights=False, temporal_layers=0,
          zero_proj_out=False):
    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                    map_location="cpu"), strict=True)
    # temporal 模块正常构建(use_temporal_module=True),但**不加载**预训练 temporal 权重 → 从零
    # ★ --temporal_layers:只覆盖**可训练的** temporal_module_kwargs(Temporal_Self),
    #   不动 motion_module_kwargs(Spatial_Cross,官方冻结权重,改了就载不进去)。
    #   不直接改 yaml,否则所有已有 checkpoint 的结构对不上。
    _uak = ic.unet_additional_kwargs
    if temporal_layers and temporal_layers != _uak.temporal_module_kwargs.num_transformer_block:
        # build() 在 main() 之外,拿不到 is_main;用 rank 判断(单机非分布式时 LOCAL_RANK 缺省为 0)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[arch] temporal_module num_transformer_block "
                  f"{_uak.temporal_module_kwargs.num_transformer_block} → {temporal_layers}"
                  f"  (权重与旧 ckpt 不兼容,必须从零训)", flush=True)
        _uak = OmegaConf.merge(_uak, {"temporal_module_kwargs": {"num_transformer_block": temporal_layers}})
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=_uak).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    # ★ 灌入 stage1 的 image backbone(不含 temporal)
    sd = torch.load(stage1_ckpt, map_location="cpu"); sd = sd.get("denoising_unet", sd)
    sd = {k: v for k, v in sd.items() if "temporal_modules" not in k}
    missing = denu.load_state_dict(sd, strict=False)
    # ★ resume:载入完整 stage2 ckpt(含已训好的 temporal),覆盖上面的随机初始化
    # ★ resume_temporal_only:**换数据后的热启动**用这个 —— 只取 temporal_modules,
    #   否则 rsd 里的空间主干会把上面刚灌入的新 stage1 主干整个覆盖掉,
    #   等于新数据上训的 stage1 完全作废(且不报任何错)。
    if resume:
        _rck = torch.load(resume, map_location="cpu")
        # ★ --resume_use_ema:取 ckpt 里的 EMA 权重而非瞬时权重做初始化。
        #   EMA 是训练轨迹的平滑平均,在 eff_bsz=4 的高梯度噪声下通常优于任一瞬时点
        #   (2026-09-18:同一 run 的 PSNR 在相邻采样点间摆动 0.5,瞬时权重挑哪个都是赌)。
        if use_ema_weights:
            assert "ema" in _rck, f"{resume} 里没有 ema 键(该 ckpt 是无 EMA 时代训的)"
            rsd = dict(_rck["denoising_unet"]); rsd.update(_rck["ema"])   # EMA 只覆盖它跟踪的那部分
        else:
            rsd = _rck.get("denoising_unet", _rck)
        if resume_temporal_only:
            rsd = {k: v for k, v in rsd.items() if "temporal_modules" in k}
        denu.load_state_dict(rsd, strict=False)
    # ★ --zero_proj_out:载入官方 temporal 后把 proj_out 清零,起点回到恒等映射。
    #   背景(2026-09-15 实测):官方 temporal 是在 ε 空间主干上训的,直接灌进 v 空间主干
    #   会一上来就往主干注入按错误尺度计算的强信号 —— step0 的 vmse 0.498(比完全没有
    #   时序模块的 0.231 还差一倍)、DYN 160×、FID 379。训 200 步能恢复成正常视频
    #   (PSNR 24.1/FID 32.9,追平随机初始化 250 步),但 DYN 仍有 5.6×,即"注意力模式对、
    #   输出增益错"。proj_out 清零后:q/k/v/o 与 norm 保留官方的运动先验(546 个张量的绝大多数),
    #   而增益从 0 自己长到合适档位 —— 既无起步灾难,也不必花几百步去压过大的增益。
    if zero_proj_out:
        n0 = 0
        with torch.no_grad():
            for nm, prm in denu.named_parameters():
                if "temporal_modules" in nm and "proj_out" in nm:
                    prm.zero_(); n0 += 1
        print(f"[zero_proj_out] 已清零 {n0} 个 temporal proj_out 张量 → 起点=恒等映射", flush=True)
    if use_rope:
        set_temporal_rope(denu, True, mode="bidir")
    imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
    return refu, denu, imgenc, len(sd)


def build_loader(args, dev, dt, rank=0, world=1):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    def make(src):
        # ★ --pose_real / --bbox_drop:见 FLOW_DISTILL_PROGRESS §5d
        # ★ --pose_real 原来把路径**写死成 hallo3 的**,任何非 hallo3 的 source 都会读错目录:
        #   clip 名对不上时报错,万一撞名则静默喂错数据。改为按 source 推导。
        _pd = str(src.pose_dir)
        _pr = _pd.replace("pose_embed", "pose_embed_real") if args.pose_real else _pd
        if args.pose_real and not os.path.isdir(_pr):
            raise FileNotFoundError(f"--pose_real 需要 {_pr},该 source 未准备 pose_embed_real")
        return MotarDataset(pose_dir=_pr,
                            pose_dir_alt=(_pd if args.pose_real else None), bbox_drop=args.bbox_drop,
                            audio_dir=src.audio_dir, caption_dir=src.caption_dir,
            data_name_path=src.data_name_path, tokenizer_path=dcfg.tokenizer_path,
            data_stats_path=dcfg.data_stats_path, context_length=args.L, fps=dcfg.fps, sr=dcfg.sr,
            text_max_len=dcfg.get("text_max_len", 128), random_crop=True, pad_short=False, load_video=True,
            latent_dir=src.latent_dir, video_dir=src.video_dir, video_processor=vproc)
    # ★ §8.0:训练集只用 hallo3(--sources 0);MEAD 仅作 eval 参考
    _srcs = list(dcfg.sources) if args.sources is None else [dcfg.sources[i] for i in args.sources]
    ds = ConcatDataset([make(s) for s in _srcs])
    sampler = None
    if world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None), num_workers=8,
                    pin_memory=True, drop_last=True, prefetch_factor=3, persistent_workers=True)
    stats = torch.load(dcfg.data_stats_path, map_location="cpu")
    return dl, stats["mean"].reshape(-1).to(dev, dt), stats["std"].reshape(-1).to(dev, dt)


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
    ap.add_argument("--grad_stat", action="store_true",
                    help="边训练边统计梯度范数与噪声(排查发散/选 batch size)")
    ap.add_argument("--val_only", default=None,
                    help="glob 模式:对匹配的每个 ckpt 跑完整验证后退出")
    ap.add_argument("--pose_real", action="store_true",
                    help="用真实逐帧 bbox 的 motion latent(pose_embed_real);\n                          背景运动本无条件信号,导致背景乱动,见 §5d")
    ap.add_argument("--bbox_drop", type=float, default=0.0,
                    help="按此概率改用常量 [0,0,1] 版,让推理时的输入也在训练分布内")
    ap.add_argument("--sources", type=int, nargs="+", default=None,
                    help="用哪些数据源(0=hallo3 1=MEAD)。§8.0 目标域为 hallo3,应传 0")
    ap.add_argument("--zero_proj_out", action="store_true",
                    help="载入 resume 的 temporal 后把 proj_out 清零(起点=恒等)。"
                         "用官方 ε 空间 temporal 热启动 v 空间主干时必开,否则起步即失控")
    ap.add_argument("--temporal_layers", type=int, default=0,
                    help="覆盖 temporal_module 的 num_transformer_block(0=用配置值)。改了就与旧 ckpt 不兼容")
    ap.add_argument("--resume_use_ema", action="store_true",
                    help="从 ckpt 的 ema 键初始化(而非瞬时权重);要求该 ckpt 是带 EMA 训的")
    ap.add_argument("--resume_temporal_only", action="store_true",
                    help="resume 只取 temporal_modules,空间主干保留 --stage1_ckpt(换数据热启动用)")
    ap.add_argument("--stage1_ckpt", required=False,
                    default=XP("XN_OUTPUT", "flow_stage1/stage1_step_2000.pt"))
    ap.add_argument("--rope", action="store_true", help="temporal 用 RoPE(推荐);否则用原加性正弦绝对 PE")
    ap.add_argument("--resume", default=None,
                    help="从完整 stage2 ckpt 续训(含已训好的 temporal);不传则 temporal 从零开始")
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-4, help="从零训新模块的常规量级(不是微调的 1e-5)")
    ap.add_argument("--lr_anneal_steps", type=int, default=0,
                    help=">0:从 --lr 余弦退火到 --lr_min,共这么多步,结束即停(max_steps 被覆盖)。"
                         "用于恒定 lr 训到平台后收割弱梯度特征(细微运动),见 HANDOFF §1")
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--t_mode", choices=["uniform"], default="uniform",
                    help="σ 分布已写死 uniform;传 lognorm 会被 argparse 直接拒绝(而非静默退回旧行为)")
    ap.add_argument("--sigma_shift", type=float, default=1.0,
                    help="训练 σ 采样的 shift(SD3/Wan 口径)。1.0=原状。★ 必须与下游"
                         "gen_ode_pairs --shift / distill --flow_grid_shift 保持一致")
    ap.add_argument("--cfg_drop", type=float, default=0.0,
                    help="★condition dropout 概率(per micro-batch)。以此概率把该 micro-batch 训成 CFG 的 uncond 形态:"
                         "reference bank 不注入(rreader.clear()) + CLIP 置零 + motion 换成参考帧(首帧)。"
                         "训练见过 uncond 后,推理端 CFG 才是分布内的。0=关闭")
    ap.add_argument("--out", default=XP("XN_OUTPUT", "flow_stage2"))
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
    # ★ EMA:整条 flow 线原来都没有,导致权重在 eff_bsz=4 下游走、指标 ±7% 振荡(见 src/utils/ema.py)
    # ★ val_window>0:验证用滑窗生成整段(训练 L 短、评测要 64 帧时必须)
    ap.add_argument("--val_window", type=int, default=0, help="0=整段单窗;>0=滑窗长度")
    ap.add_argument("--val_overlap", type=int, default=4)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="0 或负数 = 关闭 EMA。0.999 约对应 1000 步的平均窗口")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"],
                    help="影子权重放哪:auto=跟参数同卡(快 17x,多占约 1.8GB);cpu=省显存但每步多 0.25s")
    ap.add_argument("--val_sample_steps", type=int, default=20, help="验证采样步数(求快,不必等于部署步数)")

    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.batch, args.L = 20, 10000, 5, 1, 1, 32

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

    refu, denu, imgenc, n_loaded = build(dev, dt, args.stage1_ckpt, args.rope, args.resume,
                                         args.resume_temporal_only, args.resume_use_ema, args.temporal_layers, args.zero_proj_out)

    # ★ 只训 temporal_modules,image backbone 全冻结
    denu.requires_grad_(False)
    for nm, p in denu.named_parameters():
        if "temporal_modules" in nm:
            p.requires_grad_(True)
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    train_params = [p for p in denu.parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in denu.parameters() if not p.requires_grad)
    denu.train()
    if is_main:
        print(f"[stage2] image backbone 冻结={frozen/1e6:.1f}M   temporal(从零,可训)={sum(p.numel() for p in train_params)/1e6:.1f}M", flush=True)
        print(f"[init] backbone ← {args.stage1_ckpt}({n_loaded} 个张量)  "
              f"temporal={('续训 ← '+args.resume+('(仅temporal)' if args.resume_temporal_only else '(全量,会覆盖主干)')) if args.resume else '随机初始化(proj_out 零初始化→起点恒等)'}", flush=True)
        print(f"[pe] {'RoPE(相对,可外推)' if args.rope else '加性正弦绝对PE'}   L={args.L}  "
              f"world={world} batch={args.batch} accum={args.accum} → 有效bsz={args.batch*args.accum*world}", flush=True)
        print(f"[cfg_drop] {args.cfg_drop}  (>0 时按此概率训 uncond:bank不注入+CLIP置零+参考帧motion)", flush=True)

    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=args.batch, fusion_blocks="full")
    if distributed:
        denu = torch.nn.parallel.DistributedDataParallel(denu, device_ids=[lrk], output_device=lrk,
                broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
        denu_core = denu.module
    else:
        denu_core = denu

    dl, m_mean, m_std = build_loader(args, dev, dt, rank, world)
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)


    # ---- 训练内验证(CLAUDE.md §6.4) + TensorBoard
    # ★ EMA:只跟踪 requires_grad 的参数(这里=temporal_modules)
    from src.utils.ema import EMA, ema_weights
    ema = EMA(denu_core, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    if is_main:
        print(f"[ema] {'启用 decay=%.4f device=%s' % (args.ema_decay, args.ema_device) if ema else '关闭'}",
              flush=True)
    _tb = None; _val = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        _tb = SummaryWriter(os.path.join(args.out, "logs", "tb"))
    if args.val_every > 0:
        from src.utils.inproc_val import Validator
        _val = Validator(out_dir=args.out, kind="video", n_clips=args.val_clips,
                         n_frames=0,
                         sample_steps=args.val_sample_steps,
                         sample_shift=args.sigma_shift, val_cfg=args.val_cfg, val_window=args.val_window, val_overlap=args.val_overlap, is_main=is_main, tb=_tb,
                         shard=distributed)   # ★ 多卡时 clip 分到各 rank 并行,汇总到 rank0

    # ★ --val_only:对一批 ckpt 逐个跑**完整**验证,用于定稿时用更大样本量复核。
    #   走训练脚本自身的 build/Validator,保证评测前向与训练一致。
    if args.val_only:
        import glob as _g, re as _re
        cks = sorted(_g.glob(args.val_only),
                     key=lambda x: int(_re.search(r"step_(\d+)", x).group(1)))
        for ck in cks:
            st = int(_re.search(r"step_(\d+)", ck).group(1))
            _raw = torch.load(ck, map_location="cpu")
            sd = _raw.get("denoising_unet", _raw)
            denu_core.load_state_dict(sd, strict=False)
            # ★ 训练内 val 走的是 EMA 权重(见下方 ema_weights(...)),val_only 必须一致,
            #   否则跨线比较的是"EMA 权重 vs 原始权重",口径不同(DATA.md §22 同类坑)。
            if args.resume_use_ema and isinstance(_raw, dict) and "ema" in _raw:
                _e = _raw["ema"]; _e = _e.get("shadow", _e)
                _hit = denu_core.load_state_dict(_e, strict=False)
                if is_main:
                    print(f"[val_only] 已套用 EMA 权重({len(_e)} 个张量)", flush=True)
            elif is_main:
                print(f"[val_only] ⚠ 用原始权重(未套 EMA)", flush=True)
            # ★ rr/rw 必传:_set_ref 要用 reference bank 的 reader/writer,不传会 NoneType.clear()
            _val.run(st, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter, do_sample=True)
        if is_main: print("[done] val_only", flush=True)
        return
    _gn_hist = []; _micro_gn = []; _noise_hist = []; _nan_hits = 0
    step = 0; t_last = time.time(); loss_acc = 0.0; checked = False
    it = iter(dl)
    # ★ 训练前先跑一次 step 0 基线,否则看不到第一段的跃升
    if args.val_every > 0:
        # ★ shard 模式下所有 rank 都要进 run(),否则 rank0 的 all_gather 会永远等不到对端
        if is_main or distributed:
            _val.run(0, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter,
                     do_sample=args.val_sample_every > 0)
        if distributed:
            dist.barrier()

    if args.lr_anneal_steps > 0:
        args.max_steps = args.lr_anneal_steps
        if is_main:
            print(f"[lr] 余弦退火 {args.lr:g} → {args.lr_min:g},共 {args.lr_anneal_steps} 步(max_steps 已覆盖)", flush=True)
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for _ in range(args.accum):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()
            B, T = x0.shape[0], x0.shape[2]
            motion = (b["motion_tensor"].to(dev, dt) * (m_std + 1e-6) + m_mean).reshape(B, T, 32, 16)
            mask = b.get("mask", None)
            if mask is not None: mask = mask.to(dev, dt).view(B, 1, T, 1, 1)
            # ★ CFG condition dropout:以 cfg_drop 概率把本 micro-batch 训成 uncond 形态
            drop = (args.cfg_drop > 0) and (torch.rand(1).item() < args.cfg_drop)
            with torch.no_grad():
                clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
                rwriter.clear()
                refu(b["ref_latent"].to(dev, dt), torch.zeros((), device=dev).long(),
                     encoder_hidden_states=clip_emb, return_dict=False)
                rreader.update(rwriter, dtype=dt)
                if drop:
                    rreader.clear()                       # bank 置空 = 不注入参考图空间特征
                    clip_emb = torch.zeros_like(clip_emb)  # CLIP 置零
                    motion = motion[:, 0:1].expand_as(motion).contiguous()  # 参考帧(首帧)motion
            sig = sample_sigma(B, dev, args.t_mode, args.sigma_shift).view(B, 1, 1, 1, 1)
            noise = torch.randn_like(x0)
            z = ((1 - sig) * x0.float() + sig * noise.float()).to(dt)
            v_tgt = (noise.float() - x0.float())
            t_emb = (sig.view(B) * 1000.0).to(dt)
            if not checked and is_main:
                print(f"[check] x0.std={x0.float().std():.3f} v_target.std={v_tgt.std():.3f} L={T}", flush=True); checked = True
            v_pred = denu(z, t_emb, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
            se = (v_pred.float() - v_tgt) ** 2
            loss = (se * mask).sum() / mask.expand_as(se).sum().clamp_min(1) if mask is not None else se.mean()
            (loss / args.accum).backward(); micro += loss.item() / args.accum
            # ★ 逐 micro-batch 记录累积梯度的范数增量,用于估计梯度噪声:
            #   相邻 micro 之间范数抖动越大,说明单个 micro 的梯度方向越不一致(噪声越大)。
            if args.grad_stat:
                # ★ 记录**累积**梯度的平方范数 ‖G_k‖²(k=1..accum)。
                #   设 g_i = ḡ + ε_i(E[ε]=0, E‖ε‖²=σ²),则 G_k = Σ_{i≤k} g_i 满足
                #       E‖G_k‖² = k²‖ḡ‖² + k·σ²
                #   —— 关于 k 的二次式(无常数项)。对 k 做最小二乘即可同时解出
                #   ‖ḡ‖²(k² 系数)与 σ²(k 系数),无需保存 per-sample 梯度向量
                #   (1.2B 参数的 buffer 会直接 OOM)。
                with torch.no_grad():
                    _g2 = sum((p.grad.float()**2).sum() for p in train_params
                              if p.grad is not None).item()
                _micro_gn.append(_g2)
        # ★ clip_grad_norm_ 返回的是**裁剪前**的总范数,直接拿来监控发散
        _tot_gn = float(torch.nn.utils.clip_grad_norm_(train_params, 1.0))
        # ★ NaN/Inf 早期告警:发散往往先在梯度上出现,再过几步才污染到 loss。
        #   不加这个就只能看到"某一步 loss 突然变 nan",无法定位。
        if not np.isfinite(_tot_gn):
            _nan_hits += 1
            if is_main and _nan_hits <= 5:
                print(f"[!] step {step+1} 梯度范数非有限({_tot_gn}) —— 本步已跳过更新", flush=True)
            opt.zero_grad(set_to_none=True); _micro_gn = []
            step += 1; loss_acc += micro
            continue
        if args.lr_anneal_steps > 0:
            # ★ 余弦退火:lr(t) = lr_min + ½(lr − lr_min)(1 + cos πt/N),t = 已完成步数
            _t = min(step, args.lr_anneal_steps) / args.lr_anneal_steps
            _lr_now = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * _t))
            for _g in opt.param_groups:
                _g["lr"] = _lr_now
        opt.step()
        if ema is not None:
            ema.update(denu_core)
        if args.grad_stat:
            _gn_hist.append(_tot_gn)
            if len(_micro_gn) >= 3:
                _k = np.arange(1, len(_micro_gn)+1, dtype=np.float64)
                _y = np.array(_micro_gn, dtype=np.float64)
                # 拟合 y = a·k² + b·k   → a=‖ḡ‖²(真实梯度), b=σ²(单样本噪声方差)
                _A = np.stack([_k**2, _k], 1)
                _a, _b = np.linalg.lstsq(_A, _y, rcond=None)[0]
                if _a > 1e-12 and _b > 0:
                    # B_noise = σ²/‖ḡ‖²:临界 batch size。B<B_noise 时增大 batch 近似线性提速,
                    # 超过则收益递减(McCandlish et al., An Empirical Model of Large-Batch Training)
                    _noise_hist.append(float(_b / _a))
            _micro_gn = []
        step += 1; loss_acc += micro
        if step % args.log_every == 0 and is_main:
            dts = (time.time() - t_last) / args.log_every; t_last = time.time()
            _gs = ""
            if args.grad_stat and _gn_hist:
                _g = np.array(_gn_hist)
                # rel_std:梯度范数在若干步之间的相对波动,反映噪声水平(∝1/sqrt(bsz))
                _eb = args.batch*args.accum*world
                _gs = f"  |g|={_g.mean():.3f}  eff_bsz={_eb}"
                if _noise_hist:
                    _ns = float(np.median(_noise_hist))   # 用中位数,少数步的拟合会很飘
                    _gs += f"  B_noise={_ns:.1f}(当前{_eb}, {'欠' if _eb<_ns else '够'})"
                    if _tb is not None: _tb.add_scalar("grad/B_noise", _ns, step)
                    _noise_hist = []
                if _tb is not None:
                    _tb.add_scalar("grad/norm_mean", _g.mean(), step)
                    _tb.add_scalar("grad/norm_relstd", _g.std()/max(_g.mean(),1e-9), step)
                _gn_hist = []
            print(f"step {step:6d}  video_flow_mse={loss_acc/args.log_every:.5f}  {dts:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB  lr={opt.param_groups[0]['lr']:.2e}{_gs}", flush=True)
            if _tb is not None:
                _tb.add_scalar("train/loss", loss_acc / args.log_every, step)
                _tb.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
            loss_acc = 0.0
        _do_s = args.val_sample_every > 0 and step % args.val_sample_every == 0
        # ★ 采样档独立判定:val_sample_every 不必是 val_every 的倍数
        #   (曾因嵌套判定导致 --val_sample_every 250 只在 500/1000/... 触发)
        if args.val_every > 0 and (step % args.val_every == 0 or _do_s):
            if is_main or distributed:      # shard 模式:各 rank 分担 clip
                # ★ 用 EMA 权重评测:训练权重在高梯度噪声下游走,单点指标不可信
                with ema_weights(ema, denu_core):
                    _val.run(step, denu_core, refu, imgenc, dev, dt, rr=rreader, rw=rwriter,
                             do_sample=_do_s)
            if distributed:
                dist.barrier()          # 等 rank0 汇总算完 FID/FVD 再一起回到训练
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            _ck = {"denoising_unet": denu_core.state_dict(), "step": step, "args": vars(args),
                   "objective": "rectified_flow_v", "stage": 2, "rope": args.rope,
                   "cfg_drop": args.cfg_drop}
            if ema is not None:
                _ck["ema"] = ema.state_dict()      # 只含 temporal_modules,推理时覆盖同名张量即可
            torch.save(_ck, f"{args.out}/stage2_step_{step}.pt")
            print(f"[save] {args.out}/stage2_step_{step}.pt", flush=True)
            cks = sorted(glob.glob(f"{args.out}/stage2_step_*.pt"),
                         key=lambda p: int(os.path.basename(p)[len('stage2_step_'):-3]))
            for old in cks[:-args.keep_last]:
                try: os.remove(old); print(f"[rmckpt] {old}", flush=True)
                except OSError: pass
    if is_main: print("[done]")


if __name__ == "__main__":
    main()

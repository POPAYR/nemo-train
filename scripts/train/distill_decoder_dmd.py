"""
Phase 3: DMD2 蒸馏（忠实复刻 Self-Forcing native 配置）+ 工程化训练框架
======================================================================
方法（不变）：全量微调（无 LoRA）；native LRs（gen 2e-6 / critic 4e-7）；betas(0,0.999)；wd 0.01；
gen:critic=1:5；grad-clip 10.0；EMA(0.99, start 200, 部署用 EMA)；纯 DMD（无锚）。
DDPM 适配（非 flow）：ε-pred、ε-MSE critic loss、直接 timestep。

⭐ 本版对齐官方的两处 + 工程化：
  ① 长 rollout（--L 32，4 block）+ 几乎全帧梯度（--grad_frames -1 = 全部）——修 block 边界跳变；
     官方 start_gradient_frame_index = num_output_frames-21 → 标准长度下全帧有梯度。
  ② B=1 + ZeRO（ZeroRedundancyOptimizer，分片优化器态省显存）+ 大 accum。
  ③ 每个实验独立、整齐的输出目录：logs/ tensorboard/ ckpt/ samples/ + config.yaml；
     周期性可视化 test（渲染视频 mp4 + 帧条 png）；信息量丰富的带时间戳 log。

用法（单卡 smoke）:  CUDA_VISIBLE_DEVICES=5 python scripts/train/distill_decoder_dmd.py --smoke
用法（双卡）:        CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port 29570 \
                       scripts/train/distill_decoder_dmd.py --exp_name dmd2_longroll_0701 \
                       --resume output/dmd2_native_0629/dmd2_step_10000.pt \
                       --L 32 --grad_frames -1 --batch 1 --accum 8
"""
import os, sys, argparse, glob, time, logging, datetime
import torch
import torch.distributed as dist
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
sys.path.append(third_party("motar"))
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributed.optim import ZeroRedundancyOptimizer
from diffusers.video_processor import VideoProcessor
from data.dataset import MotarDataset
from src.distill.models import DMD2Models
from src.distill.dmd_step import generator_loss as eps_generator_loss, critic_loss as eps_critic_loss
from src.distill.flow_step import generator_loss as flow_generator_loss, critic_loss as flow_critic_loss
from src.distill.flow_math import flow_step_list

TRAIN_CFG = XP("REPO", "configs/train_ar.yaml")
ODE_INIT = XP("XN_OUTPUT", "ode_init/ode_step_8000.pt")
# ⚠️ 旧的 flow_teacher/flow_teacher_FINAL.pt 是 train_scope=all 练坏的模型(空间主干依赖 temporal,
#    画面有暗斑、指标被闪烁污染),已于 2026-08-19 删除。现役 teacher = stage1+stage2 两阶段产物。
FLOW_TEACHER = XP("XN_OUTPUT", "flow_stage2_cfgdrop/CUM1500.pt")
OUT_ROOT = XP("XN_OUTPUT")


# ------------------------------------------------------------------ 工程化基础设施
def setup_experiment(exp_name, rank, config_dict):
    """建立整齐的实验目录 + 返回各子目录路径（rank0 才创建/写盘）。
    output/<exp>/{logs, tensorboard, ckpt, samples}/ + config.yaml"""
    root = os.path.join(OUT_ROOT, exp_name)
    dirs = {k: os.path.join(root, k) for k in ["logs", "tensorboard", "ckpt", "samples"]}
    dirs["root"] = root
    if rank == 0:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config_dict), os.path.join(root, "config.yaml"))
    return dirs


def setup_logging(log_dir, rank):
    """带时间戳的 logger，rank0 同时写文件 + 控制台，其余 rank 只 warning。"""
    logger = logging.getLogger("dmd2")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
    log_path = None
    if rank == 0:
        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        log_path = os.path.join(log_dir, f"train_{ts}.log")
        fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def build_loader(L, batch, world=1, rank=0, sources=None, data_list=None):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    def make(s):
        return MotarDataset(pose_dir=s.pose_dir, audio_dir=s.audio_dir, caption_dir=s.caption_dir,
                            data_name_path=s.data_name_path, tokenizer_path=dcfg.tokenizer_path,
                            data_stats_path=dcfg.data_stats_path, context_length=L, fps=dcfg.fps, sr=dcfg.sr,
                            text_max_len=dcfg.get("text_max_len", 128), random_crop=True, pad_short=True,
                            load_video=True, latent_dir=s.latent_dir, video_dir=s.video_dir, video_processor=vproc)
    _srcs = list(dcfg.sources) if sources is None else [dcfg.sources[i] for i in sources]
    # ★ --data_list:覆盖清单。DMD/ODE 两阶段的 loss **不用 pad mask**,
    #   <64 帧样本会被 pad_short 末帧 repeat 成静止帧当真值学 → 传 ≥64 帧清单
    if data_list:
        _srcs = [OmegaConf.merge(s, {'data_name_path': data_list}) for s in _srcs]
    ds = ConcatDataset([make(s) for s in _srcs])
    sampler = None
    if world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=(sampler is None), num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=3)
    stats = torch.load(dcfg.data_stats_path, map_location="cpu")
    return dl, stats["mean"].reshape(-1), stats["std"].reshape(-1)


class EMA:
    """generator 的 EMA（CPU fp32 shadow），native ema_weight=0.99，仅在 gen step 更新，部署用它。"""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().float().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach().float().cpu(), alpha=1 - self.decay)
    def bf16_state(self):
        return {n: v.bfloat16() for n, v in self.shadow.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default=None, help="实验名=输出目录名；缺省则用时间戳")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--L", type=int, default=32, help="rollout 帧数（block 的整数倍）")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--window", type=int, default=0,
                    help="DMD 打分/梯度的随机连续窗帧数。**0 = 跟随 --L(全帧覆盖,推荐)**。\n"
                         "★ 旧默认值 24 来自绝对 PE 时代:当时 teacher 的 PE 上限 32 帧,窗口不能更大。\n"
                         "  换 RoPE 后该约束早已不存在(见文档 §5),继续用 24 会让 L=64 时梯度只覆盖 37.5%,\n"
                         "  且与历史成功配置都不匹配(v1: L32/win24=75%,v7: L64/win64=100%)。")
    ap.add_argument("--batch", type=int, default=1)          # 你的要求：B=1
    ap.add_argument("--accum", type=int, default=8)          # 大 accum 补偿 B=1
    ap.add_argument("--ratio", type=int, default=5)          # native dfake_gen_update_ratio
    ap.add_argument("--gen_lr", type=float, default=2e-6)
    ap.add_argument("--critic_lr", type=float, default=4e-7)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--ema_decay", type=float, default=0.99)
    ap.add_argument("--ema_start", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=0.0, help="native 3.0；需 neg_motion plumbing，暂 0")
    ap.add_argument("--zero", type=int, default=1, help="1=ZeroRedundancyOptimizer 分片优化器态省显存")
    ap.add_argument("--gen_ckpt", default=ODE_INIT)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dsl", default="999,749,499,249", help="denoising_step_list（逗号分隔）；4步=999,749,499,249；2步=999,499；1步=999")
    ap.add_argument("--reset_step", action="store_true", help="从 ckpt 热启权重但把 step 计数归零（分叉少步 run 用）")
    # ---- objective：eps(原 DDPM ε-DMD，不变) / flow(rectified-flow v-pred teacher) ----
    ap.add_argument("--objective", choices=["eps", "flow"], default="eps",
                    help="flow=接 flow teacher(需 --flow_ckpt + --gen_ckpt 用 flow 因果 init)")
    ap.add_argument("--flow_ckpt", default=FLOW_TEACHER, help="flow 模式:双向 flow teacher(灌 teacher/critic)")
    ap.add_argument("--flow_steps", type=int, default=4, help="flow 模式 student 步数(部署目标 4);σ 网格由 scheduler 生成")
    ap.add_argument("--flow_shift", type=float, default=1.0, help="FlowMatchEulerDiscreteScheduler shift(与 flow_render 一致=1.0)")
    ap.add_argument("--flow_grid_shift", type=float, default=1.0,
                    help="仅 flow_uniform_sigma=1 时生效:σ=s·t/(1+(s-1)t) 的 s。"
                         "1.0=均匀(现状)。★ 必须与生成轨迹时 gen_ode_pairs.py --shift 一致")
    ap.add_argument("--flow_uniform_sigma", type=int, default=1,
                    help="1=均匀σ网格[1.0,0.75,0.5,0.25](与 causal init 一致,官方口径);0=diffusers网格(末档空转,勿用)")
    ap.add_argument("--score_shift", type=float, default=1.0,
                    help="flow 模式 DMD打分 σ shift;1.0=不变换(实测优于官方5.0,见 diag_dmd_direction.py)")
    ap.add_argument("--sigma_high", type=float, default=0.70,
                    help="DMD打分 σ 上限。实测 σ≳0.72 时 DMD 目标比 x0 更糊(变糊区),必须截断")
    # ---- GAN（DMD2 少步锐化；官方 gan.py 迁移）----
    ap.add_argument("--reg_weight", type=float, default=0.0,
                    help="ODE 回归锚权重(DMD v1 的 L_reg,官方 0.25)。>0 时 generator 步改用 ODE 轨迹对:"
                         "同一初始噪声 rollout,回归 teacher ODE 终点。0=关闭(=DMD2 配置,已知不稳)")
    ap.add_argument("--data_list", default=None,
                    help="覆盖数据清单路径(DMD 不用 pad mask,应传 ≥64 帧清单)")
    ap.add_argument("--sources", type=int, nargs="+", default=None,
                    help="训练只用 train_ar.yaml 的第几个数据源(0=hallo3, 1=MEAD);缺省=全部。"
                         "2026-08-29 起目标域定为 hallo3,应传 --sources 0")
    ap.add_argument("--reg_micro", type=int, default=1,
                    help="每个 generator step 用几个 micro-batch 走 ODE 轨迹(算 reg+DMD),"
                         "其余走全量数据(只算 DMD)。⚠️ 设成 accum 会让 DMD 项也被限制在"
                         "2701 条轨迹上(v7 的教训:数据多样性降 78×,FID/FVD 大退)")
    ap.add_argument("--ode_pairs", default=XP("XN_OUTPUT", "ode_pairs"),
                    help="ODE 轨迹目录(gen_ode_pairs.py 产物)")
    ap.add_argument("--cfg_sigma_cut", type=float, default=1.0,
                    help="分段CFG阈值:σ>该值时teacher打分**完全关闭**CFG(1.x-Distill式4)。"
                         "1.0=关闭本功能(全σ都用CFG)。实测建议 0.75~0.8,勿照抄论文的0.94")
    ap.add_argument("--gan", action="store_true", help="开启 DMD+GAN（判别器=critic 主干+mid-block 头）")
    ap.add_argument("--gan_g_weight", type=float, default=1e-2, help="生成器对抗损失权重（官方 1e-2）")
    ap.add_argument("--gan_d_weight", type=float, default=1e-2, help="判别器损失权重（官方 1e-2）")
    ap.add_argument("--gan_start", type=int, default=0, help="从该 step 起启用 GAN（热启后先纯 DMD 预热判别器头则>0）")
    ap.add_argument("--disc_lr_mult", type=float, default=1.0, help="判别器头 lr = critic_lr×此值（官方默认 1.0；新鲜小头需大幅调高，如 100→4e-5）")
    ap.add_argument("--disc_warmup", type=int, default=0, help="D-only 预热步数：gan_start 后先只训判别器(G 对抗项关)，让 D 拉开 margin 再推 G")
    # ---- Self-Forcing paper 稳定配方（relativistic + R1/R2，小 batch 用正则替代大 batch）----
    ap.add_argument("--relativistic", action="store_true", help="相对判别器 loss（paper 全实验采用）")
    ap.add_argument("--r1r2_weight", type=float, default=0.0, help="R1/R2 有限差分正则 λ（paper 30；0=关）")
    ap.add_argument("--r1r2_sigma", type=float, default=0.05, help="R1/R2 扰动 σ（paper 0.05）")
    ap.add_argument("--reg_interval", type=int, default=1, help="每 N 个 critic step 施一次 R1/R2（lazy reg，省算力；施加时 λ×N 保持均值）")
    ap.add_argument("--val_every", type=int, default=0,
                    help="每 N 步在固定验证子集上算 x0-MSE(便宜)。0=关闭。见 CLAUDE.md §6.4")
    ap.add_argument("--val_sample_every", type=int, default=0,
                    help="每 N 步额外做因果 renoise rollout 采样,算 PSNR/SSIM/LPIPS/FID 并存图")
    ap.add_argument("--val_clips", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_last", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_frames", type=int, default=48, help="周期 test 渲染帧数（拉长看边界/漂移）")
    args = ap.parse_args()
    if args.window <= 0:
        args.window = args.L        # 默认全帧覆盖;显式传 --window N 才缩窗
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.ema_start, args.eval_every = 12, 10000, 3, 2, 4, 6
        args.eval_frames = 16
    if args.exp_name is None:
        args.exp_name = "dmd2_" + datetime.datetime.now().strftime("%m%d_%H%M")

    # ---- distributed ----
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        lrk = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrk); dev = torch.device(f"cuda:{lrk}")
    else:
        rank, world, dev = 0, 1, torch.device("cuda:0")
    is_main = (rank == 0); dt = torch.bfloat16

    # ---- 工程化：实验目录 + 日志 + tensorboard ----
    cfg_dump = {**vars(args), "world_size": world, "effective_batch": args.batch * args.accum * world}
    dirs = setup_experiment(args.exp_name, rank, cfg_dump)
    logger, log_path = setup_logging(dirs["logs"], rank)
    # step_list:eps=DDPM timesteps(int);flow=σ 网格(float,由 FlowMatchEuler scheduler 生成,去掉末尾 0)。
    # 同时绑定对应的 loss / rollout 实现(objective 分支,ε 路径行为不变)。
    if args.objective == "flow":
        if args.flow_uniform_sigma:
            # ★ 必须与 causal init 训练时的 σ 网格逐位一致,否则 train/infer 立刻错配。
            #   diffusers set_timesteps(4) 给 [1.0,0.667,0.334,0.001] —— 末档与终点重合,
            #   第 4 步空转,"4 步"实为 3 步。官方 denoising_step_list=[1000,750,500,250]
            #   即均匀 σ=[1.0,0.75,0.5,0.25],与 gen_ode_pairs.py / ode_init_causal.py 一致。
            import numpy as _np
            _t = _np.linspace(1.0, 0.0, args.flow_steps + 1)
            _gs = args.flow_grid_shift
            # shift>1:SD3/Flux 重参数化 σ=s·t/(1+(s-1)t),把步子往高噪段挤。
            # ★ 必须与 gen_ode_pairs.py 的 --shift 取同一个值,否则 train/infer 错配。
            _sg = _gs * _t / (1.0 + (_gs - 1.0) * _t) if _gs != 1.0 else _t
            dsl = [float(x) for x in _sg[:-1]]
        else:
            _ts, _sg = flow_step_list(args.flow_steps, shift=args.flow_shift)
            dsl = [float(x) for x in _sg[:-1].tolist()]      # N 个起始 σ(高→低)
        generator_loss, critic_loss = flow_generator_loss, flow_critic_loss
        from src.distill.flow_rollout import flow_rollout as _rollout_fn
    else:
        dsl = [int(x) for x in args.dsl.split(",")]
        generator_loss, critic_loss = eps_generator_loss, eps_critic_loss
        from src.distill.rollout import self_forcing_rollout as _rollout_fn
    tb = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(dirs["tensorboard"])
        logger.info("=" * 70)
        logger.info(f"实验 exp_name = {args.exp_name}")
        logger.info(f"输出目录     = {dirs['root']}")
        logger.info(f"日志文件     = {log_path}")
        logger.info(f"启动时间     = {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info(f"设备         = world={world}  B={args.batch}/GPU  accum={args.accum}  "
                    f"→ 有效 batch={cfg_dump['effective_batch']}")
        logger.info(f"rollout      = L={args.L} ({args.L//args.block} block)  block={args.block}  "
                    f"DMD窗={args.window}帧(随机连续,≤teacher24帧推理窗)")
        logger.info(f"优化         = gen_lr={args.gen_lr} critic_lr={args.critic_lr} wd={args.wd} "
                    f"clip={args.grad_clip} ratio=1:{args.ratio} ZeRO={bool(args.zero)}")
        logger.info(f"dsl({len(dsl)}步)      = {dsl}  guidance(CFG)={args.guidance}  EMA={args.ema_decay}@{args.ema_start}")
        logger.info("=" * 70)

    M = DMD2Models(dev, dt=dt, gen_ckpt=args.gen_ckpt, block_size=args.block, real_guidance_scale=args.guidance,
                   objective=args.objective, flow_ckpt=(args.flow_ckpt if args.objective == "flow" else None))
    if args.objective == "flow":
        logger.info(f"[flow] objective=flow  teacher/critic←flow_teacher  generator←{args.gen_ckpt}(因果flow init)  "
                    f"N={args.flow_steps}步  σ网格={[round(x,3) for x in dsl]}  打分σ∈[0.02,{args.sigma_high}] shift={args.score_shift}")

    # ---- 优化器：ZeRO（分片优化器态）或普通 AdamW ----
    gp = [p for p in M.generator.parameters() if p.requires_grad]
    cp = [p for p in M.critic.parameters() if p.requires_grad]
    disc_p = [p for p in M.disc_head.parameters() if p.requires_grad] if args.gan else []
    cp_all = cp + disc_p                              # 用于 allreduce / clip_grad（含判别器头）
    # 官方 gan.py：判别器(cls)头单独 param group，lr=critic_lr×disc_lr_mult（官方默认 mult=1.0）
    critic_params = ([{"params": cp, "lr": args.critic_lr},
                      {"params": disc_p, "lr": args.critic_lr * args.disc_lr_mult}]
                     if args.gan else cp)
    if args.gan:
        logger.info(f"[gan] 判别器头 disc_head 独立 param group  lr={args.critic_lr*args.disc_lr_mult:.1e}"
                    f"（mult={args.disc_lr_mult}）  g_w={args.gan_g_weight} d_w={args.gan_d_weight} start={args.gan_start}")
    if args.zero and world > 1:
        gen_opt = ZeroRedundancyOptimizer(gp, optimizer_class=torch.optim.AdamW,
                                          lr=args.gen_lr, betas=(0.0, 0.999), weight_decay=args.wd)
        critic_opt = ZeroRedundancyOptimizer(critic_params, optimizer_class=torch.optim.AdamW,
                                             lr=args.critic_lr, betas=(0.0, 0.999), weight_decay=args.wd)
        opt_kind = "ZeroRedundancyOptimizer(优化器态分片)"
    else:
        gen_opt = torch.optim.AdamW(gp, lr=args.gen_lr, betas=(0.0, 0.999), weight_decay=args.wd)
        critic_opt = torch.optim.AdamW(critic_params, lr=args.critic_lr, betas=(0.0, 0.999), weight_decay=args.wd)
        opt_kind = "AdamW(单卡/未分片)"
    logger.info(f"[opt] full-finetune  {opt_kind}  gen@{args.gen_lr} critic@{args.critic_lr}")

    # ---- resume ----
    start_step = 0; ema = None
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu")
        M.generator.load_state_dict(rk["generator"], strict=True)
        M.critic.load_state_dict(rk["critic"], strict=True)
        if args.gan and "disc_head" in rk:
            M.disc_head.load_state_dict(rk["disc_head"]); logger.info("[resume] disc_head 已载入")
        start_step = 0 if args.reset_step else rk.get("step", 0)
        logger.info(f"[resume] {args.resume} @ ckpt-step {rk.get('step',0)}"
                    f"{' → step 归零(热启少步)' if args.reset_step else ''}（gen+critic 已载入；优化器态/EMA 重置）")

    dl, mean, std = build_loader(args.L, args.batch, world, rank, sources=args.sources, data_list=args.data_list)
    mean, std = mean.to(dev, dt), std.to(dev, dt)

    def allreduce_grads(params):
        if world <= 1: return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= world

    def prep(b):
        clip = M.clip_embed(b["ref_img"])
        motion = (b["motion_tensor"].to(dev, dt) * (std + 1e-6) + mean).reshape(-1, args.L, 32, 16)
        # 真样本帧 latent（GAN 判别器用）：dataset [B,F,C,H,W] → [B,C,F,H,W]（与 rollout x0 同布局/同 SD-VAE 空间）
        real_lat = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous() if args.gan else None
        return clip, motion, b["ref_latent"].to(dev, dt), real_lat

    def fresh_noise(B):
        return torch.randn(B, 4, args.L, 64, 64, device=dev, dtype=dt)

    # ---- 周期性可视化 test：固定样本 4 步因果 rollout + TAESD 解码 → mp4 + 帧条 png ----
    ev = None
    if is_main:
        from diffusers import AutoencoderTiny
        from transformers import CLIPImageProcessor
        from PIL import Image
        from einops import rearrange
        self_forcing_rollout = _rollout_fn          # eps/flow 通用(签名兼容:M,noise,clip,motion,step_list,block_size,grad_window,full_steps)
        from src.utils.util import save_videos_grid
        etae = AutoencoderTiny.from_pretrained(XP("XN_PRETRAINED", "taesd"), torch_dtype=dt).to(dev).eval()
        ER = XP("XN_MEAD"); en = "M003_video_down_angry_level_1_001"
        _mo_full = torch.load(f"{ER}/pose_embed/{en}.pt", map_location="cpu").float()
        EF = min(args.eval_frames, _mo_full.shape[1]); EF -= EF % args.block          # block 整数倍
        epil = Image.open(f"{ER}/face_frames/{en}/000000.jpg").convert("RGB").resize((512, 512))
        ecl = M.clip_embed(CLIPImageProcessor().preprocess(epil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt))
        erl = torch.load(f"{ER}/frame_latent/{en}.pt", map_location="cpu").float()[0:1].to(dev, dt)
        emo = _mo_full[:, :EF].reshape(1, EF, 32, 16).to(dev, dt)
        egn = torch.Generator(device=dev); egn.manual_seed(1234)
        eno = torch.randn(1, 4, EF, 64, 64, generator=egn, device=dev, dtype=dt)
        ev = (etae, erl, ecl, emo, eno, EF, rearrange, Image, self_forcing_rollout, save_videos_grid, en)
        logger.info(f"[eval] 周期 test 就绪：sample={en}  帧数={EF}（block={args.block}）→ samples/step*/")

    @torch.no_grad()
    def do_eval(stp):
        etae, erl, ecl, emo, eno, EF, rearr, Img, sfr, save_vid, en = ev
        t_e = time.time()
        M.set_reference(erl, ecl, 1)
        x0, _ = sfr(M, eno, ecl, emo, dsl, block_size=args.block, grad_window=None, full_steps=True)
        v = (etae.decode(rearr(x0, "b c f h w -> (b f) c h w")).sample / 2 + 0.5).clamp(0, 1)  # [F,3,512,512]
        sdir = os.path.join(dirs["samples"], f"step_{stp:06d}"); os.makedirs(sdir, exist_ok=True)
        # mp4
        vid = v.unsqueeze(0).permute(0, 2, 1, 3, 4).float().cpu()          # [1,3,F,512,512]
        save_vid(vid, os.path.join(sdir, f"{en}_4step.mp4"), n_rows=1, fps=25)
        # 帧条 png（含 block 边界帧，便于看跳变）
        marks = sorted(set([0] + [b * args.block for b in range(1, EF // args.block)] + [EF - 1]))[:8]
        row = Img.new("RGB", (512 * len(marks), 512))
        for j, fi in enumerate(marks):
            row.paste(Img.fromarray((v[fi].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")), (512 * j, 0))
        row.save(os.path.join(sdir, f"{en}_strip.png"))
        x0std = x0.float().std().item()
        if tb is not None:
            tb.add_scalar("eval/x0_std", x0std, stp)
            tb.add_image("eval/strip", torch.from_numpy(
                __import__("numpy").array(row)).permute(2, 0, 1), stp)
        logger.info(f"[eval] step {stp}  x0std={x0std:.3f}  帧数={EF}  用时{time.time()-t_e:.1f}s  "
                    f"→ samples/step_{stp:06d}/（mp4+strip，边界帧 {marks}）")

    # ---- 训练循环 ----
    it = iter(dl); step = start_step
    t0 = time.time(); t_start = time.time()
    gl = cl = ng = 0.0; nc = 0; last_gn_g = last_gn_c = 0.0; logg = {}; logc = {}
    def nb():
        nonlocal it
        try: return next(it)
        except StopIteration: it = iter(dl); return next(it)

    # ---- ODE 轨迹对(reg 锚用):提供 (初始噪声, teacher ODE 终点, 同一条件) ----
    reg_it = None
    if args.reg_weight > 0:
        import glob as _glob
        class _ODEPairs(torch.utils.data.Dataset):
            def __init__(self, root): self.f = sorted(_glob.glob(os.path.join(root, "*.pt")))
            def __len__(self): return len(self.f)
            def __getitem__(self, i):
                d = torch.load(self.f[i], map_location="cpu")
                return {"traj": d["traj"].detach(), "clip_emb": d["clip_emb"][0].detach(),
                        "ref_latent": d["ref_latent"][0].detach(), "motion": d["motion"][0].detach()}
        _rds = _ODEPairs(args.ode_pairs)
        _rsam = None
        if world > 1:
            from torch.utils.data import DistributedSampler
            _rsam = DistributedSampler(_rds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        _rdl = DataLoader(_rds, batch_size=args.batch, sampler=_rsam, shuffle=(_rsam is None),
                          num_workers=2, pin_memory=True, drop_last=True)
        reg_it = iter(_rdl)
        logger.info(f"[reg] ODE 回归锚开启 w={args.reg_weight}  轨迹 {len(_rds)} 条 ← {args.ode_pairs}")
        logger.info(f"[reg] ⚠️ 官方用 LPIPS(像素空间),此处用 latent MSE 近似(带梯度过 VAE 太贵)")
        logger.info(f"[reg] ⚠️ 轨迹 L=64,训练 L={args.L} → 取前 {args.L} 帧;"
                    f"teacher 是双向解,其前 {args.L} 帧受后续帧影响,存在轻微目标偏置")

    def reg_batch():
        nonlocal reg_it
        try: rb = next(reg_it)
        except StopIteration:
            reg_it = iter(_rdl); rb = next(reg_it)
        tr = rb["traj"].to(dev, dt)                       # [B,5,C,F,H,W]
        Lc = args.L
        return (tr[:, 0, :, :Lc], tr[:, -1, :, :Lc],      # 初始噪声, teacher ODE 终点
                rb["clip_emb"].to(dev, dt), rb["motion"][:, :Lc].to(dev, dt),
                rb["ref_latent"].to(dev, dt))

    _val = None            # 训练内验证器,首次用到时懒构造(需要 dsl / M.gen_causal)
    logger.info(f"[train] 开始训练：step {start_step} → {args.max_steps}")
    while step < args.max_steps:
        # ---- critic（每步，accum 个 micro）----
        gan_on = args.gan and step >= args.gan_start                       # D 对抗项开
        g_gan_on = gan_on and step >= args.gan_start + args.disc_warmup    # G 对抗项开（D 预热后）
        critic_opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            b = nb(); clip, motion, ref, real_lat = prep(b); B = clip.shape[0]
            M.set_reference(ref, clip, B)
            # lazy R1/R2：每 reg_interval 步施一次，施加时 λ×interval 保持时间均值（StyleGAN2 式）
            reg_w = (args.r1r2_weight * args.reg_interval) if (gan_on and args.r1r2_weight > 0
                     and step % args.reg_interval == 0) else 0.0
            lc, logc = critic_loss(M, fresh_noise(B), clip, motion, dsl, block_size=args.block, grad_window=args.window,
                                   real_latent=real_lat if gan_on else None,
                                   gan_d_weight=args.gan_d_weight if gan_on else 0.0,
                                   relativistic=args.relativistic, r1r2_weight=reg_w, r1r2_sigma=args.r1r2_sigma,
                                   **({"score_shift": args.score_shift, "sigma_high": args.sigma_high} if args.objective == "flow" else {}))
            (lc / args.accum).backward(); cl += lc.item() / args.accum
        allreduce_grads(cp_all); last_gn_c = torch.nn.utils.clip_grad_norm_(cp_all, args.grad_clip).item(); critic_opt.step()
        nc += 1

        # ---- generator（每 ratio 步）----
        if step % args.ratio == 0:
            gen_opt.zero_grad(set_to_none=True)
            _reg_sum, _reg_n = 0.0, 0
            for _mi in range(args.accum):
                # ★ 只有前 reg_micro 个 micro-batch 走 ODE 轨迹(reg+DMD 共用一次 rollout);
                #   其余走全量数据只算 DMD —— 否则 DMD 的分布匹配会被限制在 2701 条轨迹上,
                #   数据多样性骤降 78× → FID/FVD 大退(v7 实测 FVD +46%)。
                if args.reg_weight > 0 and _mi < args.reg_micro:
                    _no, _tgt, clip, motion, ref = reg_batch(); B = clip.shape[0]; _rl = None
                else:
                    b = nb(); clip, motion, ref, _rl = prep(b); B = clip.shape[0]
                    _no, _tgt = fresh_noise(B), None
                M.set_reference(ref, clip, B)
                lg, logg = generator_loss(M, _no, clip, motion, dsl,
                                          block_size=args.block, grad_window=args.window,
                                          guidance_scale=args.guidance,
                                          gan_g_weight=args.gan_g_weight if g_gan_on else 0.0,
                                          real_latent=_rl if g_gan_on else None,
                                          relativistic=args.relativistic,
                                          **({"score_shift": args.score_shift, "sigma_high": args.sigma_high,
                                              "cfg_sigma_cut": args.cfg_sigma_cut,
                                              "reg_target": _tgt,
                                              "reg_weight": (args.reg_weight * args.accum / max(args.reg_micro,1)
                                                             if _tgt is not None else 0.0)}
                                             if args.objective == "flow" else {}))
                if "reg" in logg: _reg_sum += logg["reg"]; _reg_n += 1
                (lg / args.accum).backward(); gl += lg.item() / args.accum
            allreduce_grads(gp); last_gn_g = torch.nn.utils.clip_grad_norm_(gp, args.grad_clip).item(); gen_opt.step(); ng += 1
            if is_main and step >= args.ema_start:
                if ema is None: ema = EMA(M.generator, args.ema_decay)
                ema.update(M.generator)

        step += 1

        # ---- 日志（信息量丰富）----
        if step % args.log_every == 0 and is_main:
            dts = (time.time() - t0) / args.log_every; t0 = time.time()
            g_avg = gl / max(ng, 1); c_avg = cl / max(nc, 1); x0s = logg.get("x0_std", 0)
            mem = torch.cuda.max_memory_allocated() / 1e9
            eta_h = (args.max_steps - step) * dts / 3600
            gan_str = ""
            if gan_on:
                wu = "" if g_gan_on else "[D预热]"
                reg_s = f" reg {logc.get('gan_reg',0):.3f}" if args.r1r2_weight > 0 else ""
                gan_str = (f" | {wu}Dr {logc.get('d_real',0):+.2f} Df {logc.get('d_fake',0):+.2f} "
                           f"Δ{logc.get('d_real',0)-logc.get('d_fake',0):+.2f} Ggan {logg.get('gan_g',0):.3f}{reg_s}")
            # 分段CFG:打出本 step 实际开了 CFG 的样本比例,确认掩码没有静默失效
            seg_str = f" | cfgOn {logg.get('cfg_frac', 1.0):.2f}" if args.cfg_sigma_cut < 1.0 else ""
            # ★ 跨 micro-batch 累计:reg_micro<accum 时最后一个 micro 没有 reg 键,
            #   只取 logg 会恒为 0(v8 踩过),必须累计。n 也打出来确认锚真的生效。
            seg_str += (f" | reg {_reg_sum/max(_reg_n,1):.4f}(n={_reg_n})"
                        if args.reg_weight > 0 else "")
            logger.info(f"step {step:6d}/{args.max_steps} | G {g_avg:.4f} | C {c_avg:.4f} | x0std {x0s:.3f} | "
                        f"gnG {last_gn_g:.2f} gnC {last_gn_c:.2f}{gan_str}{seg_str} | {dts:.2f}s/it | mem {mem:.1f}GB | ETA {eta_h:.1f}h")
            if tb is not None:
                for k, vv in [("loss/generator", g_avg), ("loss/critic", c_avg), ("stat/x0_std", x0s),
                              ("stat/grad_norm_gen", last_gn_g), ("stat/grad_norm_critic", last_gn_c),
                              ("perf/sec_per_it", dts), ("perf/mem_gb", mem)]:
                    tb.add_scalar(k, vv, step)
                if gan_on:
                    for k, kk in [("gan/d_real", "d_real"), ("gan/d_fake", "d_fake"), ("gan/d_loss", "gan_d"),
                                  ("gan/reg", "gan_reg")]:
                        tb.add_scalar(k, logc.get(kk, 0), step)
                    tb.add_scalar("gan/d_margin", logc.get("d_real", 0) - logc.get("d_fake", 0), step)
                    tb.add_scalar("gan/g_loss", logg.get("gan_g", 0), step)
            gl = cl = ng = 0.0; nc = 0

        # ---- 存档 ----
        # 注：ZeRO-1(ZeroRedundancyOptimizer) 只分片优化器态、不分片参数 → rank0 的 model.state_dict() 本就完整；
        # 且本 ckpt 不存优化器态，故无需 consolidate_state_dict（它是集合通信，只在 rank0 调会死锁→NCCL 超时 SIGABRT）。
        _do_s = args.val_sample_every > 0 and step % args.val_sample_every == 0
        # ★ 采样档独立判定:val_sample_every 不必是 val_every 的倍数
        if args.val_every > 0 and (step % args.val_every == 0 or _do_s):
            if is_main:
                if _val is None:
                    from src.utils.inproc_val import Validator
                    # ★ ctrl 用 M.gen_causal;bank 走 M.set_reference(三个 reader 统一分发)
                    #   student_sigmas = dsl,与训练档逐位一致
                    _val = Validator(out_dir=dirs["root"], kind="causal", n_clips=args.val_clips,
                                     student_sigmas=dsl, block=args.block, ctrl=M.gen_causal,
                                     set_ref_fn=lambda r, c: M.set_reference(r, c, 1),
                                     is_main=True, tb=tb, clip_len=args.block * 8)
                _val.run(step, M.generator, None, None, dev, dt, logger=logger,
                         do_sample=_do_s)
            if world > 1: dist.barrier()
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            ck = {"generator": M.generator.state_dict(), "critic": M.critic.state_dict(),
                  "step": step, "args": vars(args)}
            if args.gan: ck["disc_head"] = M.disc_head.state_dict()
            if ema is not None: ck["generator_ema"] = ema.bf16_state()
            cpath = os.path.join(dirs["ckpt"], f"dmd2_step_{step}.pt")
            torch.save(ck, cpath)
            logger.info(f"[save] step {step} → {cpath}  ({os.path.getsize(cpath)/1e9:.1f}GB)")
            for old in sorted(glob.glob(os.path.join(dirs["ckpt"], "dmd2_step_*.pt")),
                              key=lambda p: int(os.path.basename(p)[len('dmd2_step_'):-3]))[:-args.keep_last]:
                try: os.remove(old); logger.info(f"[save] 清理旧 ckpt {os.path.basename(old)}")
                except OSError: pass

        # ---- 周期性可视化 test ----
        if step % args.eval_every == 0:
            if ev is not None: do_eval(step)
            if distributed: dist.barrier()

    if is_main:
        logger.info(f"[done] 训练完成：{step} steps，总用时 {(time.time()-t_start)/3600:.1f}h")
        if tb is not None: tb.close()


if __name__ == "__main__":
    main()

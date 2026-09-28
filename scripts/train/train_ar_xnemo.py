"""AR motion 训练：video diffusion loss + 运动回归 loss + (可选) GAN adv loss 联合优化。

结构：
  - rollout_lcm     : 全程 student-forcing 的 LCM rollout（帧间 detach，O(1) 显存）。
  - Model           : G 侧。AR 出 motion_emb（归一化空间）→ denorm 喂冻结 UNet → (model_pred, motion_emb_norm)。
  - XNemoARTrainer  : 训练器。Phase0 D-warmup → Phase1+ 正常 video + adv 联合训练。

关键设计：
  - UNet / 各 encoder 全冻结，只训 AR（motion_predictor）。
  - D 移出 DeepSpeed，普通 fp32 AdamW，保住 spectral-norm parametrization。
  - disc_inst_norm：D 输入 per-sample 标准化，掐掉「整体方差」判别捷径（开启后 D 须重新 warmup）。
  - adv ramp 区间应与 video 降权区间对齐：adv 上来时 video 退下去，二者不正交对抗，GAN 才稳。
"""

import argparse
import datetime
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import deepspeed
import deepspeed.comm as dist


# --------------------------------------------------------------------------- #
# 控制台输出同时写入 log 文件（每个 rank 各一份）
# --------------------------------------------------------------------------- #
class _Tee:
    """把 stdout/stderr 同时写到终端和文件。"""

    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file = file_handle

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


# --------------------------------------------------------------------------- #
# repo 路径注入（在 import 业务模块前）
# --------------------------------------------------------------------------- #
for _env in ("AR_REPO_ROOT", "XNEMO_REPO_ROOT"):
    _p = os.environ.get(_env, "")
    if _p and _p not in sys.path:
        sys.path.append(_p)

from model.armodel import MotionTransformer, SelfAttnKVCache, CrossAttnKVCache
from data.dataset import MotarDataset
from diffusers.video_processor import VideoProcessor
from diffusers import DDPMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from transformers import UMT5EncoderModel, Wav2Vec2Model, CLIPVisionModelWithProjection

from src.losses import (
    build_mouth_weight_map,
    ddpm_video_loss,
    motion_regression_loss,
    std_ratio,
)

# 复用 GAN 阶段的判别器。
# NOTE: 路径建议改为 env / config 配置，避免换机器硬编码失效。
_DISC_REPO = os.environ.get("MOTION_DISC_ROOT", "/media/ps/ssd5/ayr/motar/model")
if _DISC_REPO not in sys.path:
    sys.path.append(_DISC_REPO)
from motion_discriminator import MotionDiscriminator, count_params

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
# loss 权重 ramp 调度
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# loss 权重 ramp 调度
# --------------------------------------------------------------------------- #
def make_ramp(spec, default_const=1.0):
    """step -> weight 调度函数。spec 支持三种形式：

    1) None / 标量          : 常数。
    2) {start_val, end_val, warmup_start, warmup_end} : 单段线性 ramp（向后兼容）。
    3) {keypoints: [[step, val], ...]} : 多段分段线性折线。step 须升序；
       区间外取端点值（左端常数、右端常数），区间内线性插值。
       用于 video / adv 错峰调度（video 先升后降、adv 后进）。
    """
    if spec is None:
        return lambda s: default_const
    if not isinstance(spec, dict) and not hasattr(spec, "get"):
        v = float(spec)
        return lambda s: v

    # ---- 形式 3：keypoints 折线 ----
    kps = spec.get("keypoints", None)
    if kps is not None:
        pts = [(int(s), float(v)) for s, v in kps]
        pts.sort(key=lambda x: x[0])
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]

        def fn_kp(step):
            if step <= steps[0]:
                return vals[0]
            if step >= steps[-1]:
                return vals[-1]
            # 找所在区间（点数通常很少，线性扫描足够）
            for i in range(1, len(steps)):
                if step <= steps[i]:
                    s0, s1 = steps[i - 1], steps[i]
                    v0, v1 = vals[i - 1], vals[i]
                    r = (step - s0) / max(1, (s1 - s0))
                    return v0 + (v1 - v0) * r
            return vals[-1]

        return fn_kp

    # ---- 形式 2：单段线性 ----
    sv = float(spec.get("start_val", default_const))
    ev = float(spec.get("end_val", default_const))
    ws = int(spec.get("warmup_start", 0))
    we = int(spec.get("warmup_end", 0))

    def fn(step):
        if step <= ws:
            return sv
        if step >= we or we <= ws:
            return ev
        r = (step - ws) / max(1, (we - ws))
        return sv + (ev - sv) * r

    return fn


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #
def normalize_wav(wav):
    return (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-7)


def align_to_frames(feat, T):
    feat = feat.transpose(1, 2)
    feat = F.interpolate(feat.float(), size=T, mode="linear", align_corners=False)
    return feat.transpose(1, 2)


def build_local_window(frame_feat, half=2):
    feat = frame_feat.transpose(1, 2)
    feat = F.pad(feat, (half, half), mode="replicate")
    windows = feat.unfold(-1, 2 * half + 1, 1)
    return windows.permute(0, 2, 3, 1).contiguous()


# --------------------------------------------------------------------------- #
# GAN losses（non-saturating logistic）
# --------------------------------------------------------------------------- #
def ns_d_loss(logits_real, logits_fake, mask):
    m = mask.unsqueeze(-1).float()
    n = m.sum().clamp(min=1)
    l_real = (F.softplus(-logits_real) * m).sum() / n
    l_fake = (F.softplus(logits_fake) * m).sum() / n
    return 0.5 * (l_real + l_fake)


def ns_g_loss(logits_fake, mask):
    m = mask.unsqueeze(-1).float()
    n = m.sum().clamp(min=1)
    return (F.softplus(-logits_fake) * m).sum() / n


# --------------------------------------------------------------------------- #
# LCM one-step diff head
# --------------------------------------------------------------------------- #
def lcm_1step(diff_net, z, alphas_cp, T_max, noise=None):
    B, D = z.shape
    if noise is None:
        noise = torch.randn(B, D, device=z.device, dtype=z.dtype)
    t = torch.full((B,), T_max - 1, device=z.device)
    v = diff_net(noise, t.float(), c=z)
    a = alphas_cp[t].view(-1, 1).to(z.dtype)
    return a.sqrt() * noise - (1 - a).sqrt() * v


# --------------------------------------------------------------------------- #
# 单帧 AR step（抽成独立函数，便于 torch.compile）。移植自 S4(sf_gan) 的 _ar_step_impl。
# 内部不含任何 DDP collective；compile 是语义保持的，fullgraph=False 下遇阻自动回退 eager（不崩）。
# 缺点：KV cache 是增长的(动态形状)，CUDA Graph(reduce-overhead) 抓不住 → 现在只能 default 模式、收益有限；
#       真正大提速要把 cache 改成静态预分配(后续单独做并验证)。
# --------------------------------------------------------------------------- #
def _ar_step_impl(cur, fusion_latents, local_t, layers, self_caches,
                  fusion_caches, motion_proj, norm, max_len):
    h = motion_proj(cur)
    for layer, sc, fc in zip(layers, self_caches, fusion_caches):
        h = layer.forward_cached(h, fusion_latents, local_t, sc, fc, max_len)
    return norm(h[:, -1])


# module-level 持有，main 里按 config 决定是否替换成 torch.compile 版（只编译一次，避免抖动）。
_AR_STEP = _ar_step_impl


def rollout_lcm(student, motion_gt_norm, text_emb, audio_emb,
                local_audio_full, alphas_cp, T_max, autocast_factory):
    """全程 student-forcing 的 LCM rollout。

    帧间 detach（O(1) 显存），每帧 motion 计算 with-grad。
    返回 preds [B, T-1, D]（归一化空间）。
    """
    B, T, D = motion_gt_norm.shape
    with autocast_factory():
        fusion_latents = student.fusion_net(text_emb, audio_emb).detach()
        local_proj = student.audio_proj(local_audio_full).detach()
        self_caches = [SelfAttnKVCache() for _ in student.layers]
        fusion_caches = [CrossAttnKVCache() for _ in student.layers]
        cur = motion_gt_norm[:, 0:1].detach()
        preds_list = []
        for t in range(T - 1):
            local_t = local_proj[:, t:t + 1]
            z = _AR_STEP(cur, fusion_latents, local_t, student.layers,
                         self_caches, fusion_caches,
                         student.motion_proj, student.norm, student.max_len)
            pred = lcm_1step(student.diffloss.net, z, alphas_cp, T_max)
            preds_list.append(pred)
            if t < T - 2:
                cur = pred.detach().unsqueeze(1)
            for sc in self_caches:
                if sc.k is not None:
                    sc.k = sc.k.detach()
                    sc.v = sc.v.detach()
            for fc in fusion_caches:
                if fc.k is not None:
                    fc.k = fc.k.detach()
                    fc.v = fc.v.detach()
    return torch.stack(preds_list, dim=1)


# =========================================================================== #
# G 侧模型
# =========================================================================== #
class Model(nn.Module):
    """AR rollout 出 motion_emb（归一化空间）→ denorm 喂冻结 UNet → (model_pred, motion_emb_norm)。

    motion_emb_norm 同时供 reg loss 与 GAN 判别器使用（D 在归一化空间预训练）。
    """

    def __init__(self, reference_unet, denoising_unet, motion_predictor,
                 ref_writer, ref_reader, alphas_cp, T_max,
                 motion_token_shape=(32, 16), mean=None, std=None, gen_norm=False):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.motion_predictor = motion_predictor
        self.ref_writer = ref_writer
        self.ref_reader = ref_reader
        self.motion_token_shape = tuple(motion_token_shape)
        self.gen_norm = gen_norm
        self.register_buffer("alphas_cp", alphas_cp, persistent=False)
        self.T_max = T_max
        if mean is not None:
            self.register_buffer("mean", mean.reshape(1, 1, -1), persistent=False)
            self.register_buffer("std", std.reshape(1, 1, -1), persistent=False)

    def _denorm(self, x):
        return x * (self.std + 1e-6) + self.mean

    def forward(self, text_emb, audio_emb, local_audio_emb, clip_img_emb,
                ref_image_latents, noisy_latents, timesteps, gt_motion_norm,
                skip_unet=False, amp_dtype=None):
        # B,T 从 motion 取（video=0/skip_unet 时 noisy_latents 为 None）；与 video 路径下 frames 对齐。
        B, T = gt_motion_norm.shape[0], gt_motion_norm.shape[1]
        if amp_dtype is None:
            amp_dtype = noisy_latents.dtype

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            preds = rollout_lcm(
                self.motion_predictor, gt_motion_norm, text_emb, audio_emb,
                local_audio_emb, self.alphas_cp, self.T_max,
                autocast_factory=lambda: torch.cuda.amp.autocast(dtype=amp_dtype),
            )
            seed = gt_motion_norm[:, 0:1].detach()
            motion_emb_norm = torch.cat([seed, preds], dim=1)  # [B,T,D] 归一化空间

        # skip_unet：video 关闭时只出 motion（给 reg/GAN），跳过冻结 UNet → 省显存/提速，可上长 context。
        if skip_unet:
            return None, motion_emb_norm

        # 喂 UNet 需要 raw 空间（除非 gen_norm 表示已是该空间）
        motion_for_unet = self._denorm(motion_emb_norm) if not self.gen_norm else motion_emb_norm
        motion_tokens = motion_for_unet.reshape(B, T, *self.motion_token_shape)

        with torch.no_grad():
            self.reference_unet(ref_image_latents, torch.zeros_like(timesteps),
                                encoder_hidden_states=clip_img_emb, return_dict=False)
            self.ref_reader.update(self.ref_writer)

        model_pred = self.denoising_unet(
            noisy_latents, timesteps,
            encoder_hidden_states=[clip_img_emb, motion_tokens],
            pose_cond_fea=None, return_dict=False)[0]

        self.ref_reader.clear()
        self.ref_writer.clear()
        # 返回归一化空间 motion 给 reg / GAN（判别器在归一化空间预训练）
        return model_pred, motion_emb_norm


# =========================================================================== #
# 训练器
# =========================================================================== #
class XNemoARTrainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda", args.local_rank)

        self._setup_output_dir(config)
        self._setup_logging()
        self._read_config(config)
        self._load_stats(config)

        print("init model...")
        self._init_models()
        print("init deepspeed...")
        self._setup_deepspeed()
        print("preparing data...")
        self._prepare_data()
        print("init visualizer...")
        self._setup_visualiser()

    # ----------------------------- setup ---------------------------------- #
    def _setup_output_dir(self, config):
        # output_dir 带时间戳：各 rank 的 datetime.now() 不同，用 rank0 广播统一。
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dist.is_initialized():
            ts_t = torch.tensor([int(ts)], device=self.device)
            dist.broadcast(ts_t, src=0)
            ts = str(int(ts_t.item())).zfill(15)
        self.output_dir = f"{config.working_dir}/exp_{ts}"
        self.ckpt_dir = Path(self.output_dir, "checkpoints")
        self.vis_dir = Path(self.output_dir, "vis")
        self.log_dir = Path(self.output_dir, "logs")
        if self.is_main():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(self.output_dir, "config.yaml"))
            self.writer = SummaryWriter(log_dir=f"{self.output_dir}/tb_logs")
        if dist.is_initialized():
            dist.barrier()
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_logging(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._log_fh = open(self.log_dir / f"console_rank{rank}.log", "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, self._log_fh)
        sys.stderr = _Tee(sys.__stderr__, self._log_fh)
        print(f"[Log] rank {rank} console -> {self.log_dir / f'console_rank{rank}.log'}")

    def _read_config(self, config):
        self.spf = config.data.sr // config.data.fps
        self.window_frames = config.data.get("window_frames", 5)
        self.vae_scale = config.get("vae_scale", 0.18215)
        self.motion_token_shape = tuple(config.get("motion_token_shape", [32, 16]))
        self.gen_norm = config.get("generate_returns_normalized", False)
        self.log_every = int(config.training.get("log_interval", 50))

        # ---- loss 权重 ramp ----
        loss_cfg = config.loss

        def _ramp(key, default):
            v = loss_cfg.get(key, default)
            v = OmegaConf.to_container(v) if OmegaConf.is_config(v) else v
            dc = float(default) if not isinstance(default, dict) else 1.0
            return make_ramp(v, default_const=dc)

        # 建议把 lambda_video 配成 high->low 的 ramp（dict）：adv 上来时 video 退下去。
        self.lam_video_fn = _ramp("lambda_video", 1.0)
        self.lam_reg_fn = _ramp("lambda_reg", 0.1)
        self.lam_adv_fn = _ramp("lambda_adv", 0.0)

        # ---- GAN 配置 ----
        gcfg = config.get("gan", {})
        self.gan_enabled = bool(gcfg.get("enabled", False))
        self.w_win = float(gcfg.get("w_win", 0.5))
        self.d_grad_clip = float(gcfg.get("d_grad_clip", 1.0))
        self.gan_window_size = int(gcfg.get("window_size", 8))
        self.disc_lr = float(gcfg.get("disc_lr", 1e-5))
        # D 输入实例标准化开关（掐掉方差捷径）。默认开启。
        self.disc_inst_norm = bool(gcfg.get("disc_inst_norm", True))
        # R1（默认关；非崩溃主因，但量级可达数百，慎用）
        self.r1_gamma = float(gcfg.get("r1_gamma", 0.0))
        self.r1_every = int(gcfg.get("r1_every", 16))
        # D-warmup
        self.d_warmup_steps = int(gcfg.get("d_warmup_steps", 1500))
        self.dwarm_min_steps = int(gcfg.get("dwarm_min_steps", 200))
        self.dwarm_gap_target = float(gcfg.get("dwarm_gap_target", 1.0))
        self.dwarm_ema_alpha = float(gcfg.get("dwarm_ema_alpha", 0.05))
        self.d_warmup_lr_boost = float(gcfg.get("d_warmup_lr_boost", 5.0))
        # 撤 boost 后不要断崖：在 d_post_warmup_anneal 步内把 lr 倍率从 boost 线性退回 1x。
        self.d_post_warmup_anneal = int(gcfg.get("d_post_warmup_anneal", 0))
        # 正常期 D:G 更新比例（D 每个 G-step 多走几个梯度块）。失稳时 >1 给 D 更多步。
        self.d_ratio = max(1, int(gcfg.get("d_ratio", 1)))
        # 自适应刹车：正常期 EMA gap 跌破此值时临时停 adv，让 D 追上来（<=0 关闭）。
        self.adv_gap_floor = float(gcfg.get("adv_gap_floor", 0.0))
        # adv 软启动：刹车解除后的前 adv_soft_steps 步，lam_adv 线性从 0 恢复到满值，
        # 避免一解除就猛压把 D 再次打穿（减小锯齿振幅）。<=0 关闭。
        self.adv_soft_steps = int(gcfg.get("adv_soft_steps", 0))
        self._adv_resume_step = -1   # 最近一次刹车解除的 step（软启动锚点）
        self._adv_braked_prev = False
        self._ema_d_gap = 0.0
        self._dwarm_done = False
        self._dwarm_done_step = 0     # warmup 实际结束的 global_step，用于 lr 退火锚点
        self._d_micro = 0
        self.gas = int(config.get("gradient_accumulation_steps", 8))

        # 隔离测试开关：adv 块照常执行（所有副作用保留），但 adv 对 G 梯度贡献清零。
        self.adv_grad_off = bool(config.get("adv_grad_off", False))
        if self.adv_grad_off and self.is_main():
            print("[ISO] ADV_GRAD_OFF=True：adv 块执行但梯度清零（隔离测试模式）")

        # video 路径总开关：True 时全程跳过冻结 UNet（不加载 video latent、不算 video loss），
        # 只跑 motion-space rollout+reg+GAN。用于纯 motion/GAN 调试，省显存→可上长 context(128)。
        self.skip_video = bool(config.get("disable_video_path", False))
        if self.skip_video and self.is_main():
            print("[skip_video=ON] 跳过 UNet/video：纯 motion-space 训练（load_video=False, l_video=0）")

    def _load_stats(self, config):
        stats = torch.load(config.data.data_stats_path, map_location="cpu")
        self.mean = stats["mean"].float().reshape(-1).to(self.device)
        self.std = stats["std"].float().reshape(-1).to(self.device)

    # ----------------------------- utils ---------------------------------- #
    def is_main(self):
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def _dbg(self, step):
        """是否在本步打印监控（仅 main rank，每 log_every 步一次）。"""
        return self.is_main() and (step % self.log_every == 0)

    @staticmethod
    def _inst_norm_motion(x, eps=1e-5):
        """实例标准化：per-sample 去均值除标准差，跨 (T, D)。掐掉 D 的整体方差判别捷径。"""
        mu = x.mean(dim=(1, 2), keepdim=True)
        sd = x.std(dim=(1, 2), keepdim=True).clamp(min=eps)
        return (x - mu) / sd

    def _disc_in(self, motion):
        """D 输入预处理：可选实例标准化。集中一处，保证 real/fake/R1 三处一致。"""
        if self.disc_inst_norm:
            return self._inst_norm_motion(motion)
        return motion

    # ----------------------------- model ----------------------------------- #
    def _init_models(self):
        cfg = self.config
        infer_config = OmegaConf.load(cfg.inference_config)

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, subfolder="unet").to(self.device)
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs).to(self.device)

        self.denoising_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
        self.reference_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                       map_location="cpu"), strict=True)
        if cfg.get("temporal_module_path", None):
            self.denoising_unet.load_state_dict(
                torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path).to(self.device)
        self.text_encoder = UMT5EncoderModel.from_pretrained(cfg.text_encoder_path).to(self.device)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.audio_encoder_path).to(self.device)
        for enc in (self.image_encoder, self.text_encoder, self.audio_encoder):
            enc.requires_grad_(False).eval()
        self.text_dim = self.text_encoder.config.d_model
        self.audio_dim = self.audio_encoder.config.hidden_size

        self.motion_predictor = MotionTransformer(
            **cfg.model, pretrained_path=cfg.pretrained_motion_predictor_path).to(self.device)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=8)

        self.alphas_cp = self.motion_predictor.diffloss.train_scheduler.alphas_cumprod.to(
            self.device).float()
        self.T_max = self.motion_predictor.diffloss.train_scheduler.config.num_train_timesteps

        # ---- resume ----
        self._resume_step, self._resume_epoch, self._resume_lr = 0, 0, None
        resume_ar = cfg.get("resume_ar_path", None)
        if resume_ar and os.path.exists(resume_ar):
            ck = torch.load(resume_ar, map_location="cpu")
            sd = ck.get("motion_predictor") or ck.get("state_dict")
            self.motion_predictor.load_state_dict(sd, strict=True)
            self._resume_step = ck.get("step", 0)
            self._resume_epoch = ck.get("epoch", 0)
            self._resume_lr = ck.get("lr_scheduler", None)
            if self.is_main():
                print(f"[Resume] AR-only from {resume_ar} @ step {self._resume_step}")

        self.ref_writer = ReferenceAttentionControl(
            self.reference_unet, do_classifier_free_guidance=False,
            mode="write", batch_size=cfg.data.batch_size, fusion_blocks="full")
        self.ref_reader = ReferenceAttentionControl(
            self.denoising_unet, do_classifier_free_guidance=False,
            mode="read", batch_size=cfg.data.batch_size, fusion_blocks="full")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        self.train_scheduler = DDPMScheduler(**sched_kwargs)
        self.num_train_timesteps = self.train_scheduler.config.num_train_timesteps
        self.pred_type = self.train_scheduler.config.prediction_type

        # ---- 只训 AR；UNet 冻结 ----
        self.trainable_params = []
        self.motion_predictor.train().requires_grad_(True)
        self.trainable_params += list(self.motion_predictor.parameters())
        self.reference_unet.requires_grad_(False).eval()
        self.denoising_unet.requires_grad_(False)
        dn_modules = list(cfg.training.get("denoising_train_modules", []))
        if dn_modules == ["all"]:
            self.denoising_unet.train().requires_grad_(True)
            self.trainable_params += list(self.denoising_unet.parameters())
        elif len(dn_modules) > 0:
            self.denoising_unet.train()
            for n, p in self.denoising_unet.named_parameters():
                if any(mm in n for mm in dn_modules):
                    p.requires_grad = True
                    self.trainable_params.append(p)
        else:
            self.denoising_unet.eval()

        self.model = Model(self.reference_unet, self.denoising_unet, self.motion_predictor,
                           self.ref_writer, self.ref_reader, self.alphas_cp, self.T_max,
                           self.motion_token_shape, mean=self.mean, std=self.std,
                           gen_norm=self.gen_norm)

        self._init_discriminator(cfg)

        if self.is_main():
            n_par = sum(p.numel() for p in self.trainable_params) / 1e6
            print(f"G trainable: {n_par:.2f}M | denoising modules: {dn_modules or 'NONE'}")
            if self.gan_enabled:
                print(f"D total: {count_params(self.disc):.2f}M | GAN ENABLED")
            print(f"gen_norm={self.gen_norm}")

    def _init_discriminator(self, cfg):
        self.disc = None
        self._disc_resumed = False    # D 是否来自「同一 run」的 ckpt（已适配当前序列长度 & inst_norm）
        self._disc_prewarmed = False  # D 是否来自 disc_warmed_ckpt（已 warmup 好）→ 跳过 warmup
        if not self.gan_enabled:
            return
        gcfg = cfg.gan
        self.disc = MotionDiscriminator(
            motion_dim=cfg.model.motion_dim, text_dim=self.text_dim, audio_dim=self.audio_dim,
            d_dim=gcfg.d_dim, heads=gcfg.d_heads, dim_head=gcfg.d_dim_head,
            depth=gcfg.d_depth, window_size=gcfg.window_size,
            cond_drop_p=gcfg.get("cond_drop_p", 0.0),
            use_spectral_norm=bool(gcfg.get("use_spectral_norm", True)),
        ).to(self.device)

        # D 权重来源优先级：
        #   1) disc_warmed_ckpt —— 之前某次 run 在「当前序列长度 & inst_norm」上 warmup 完成时存的 D，
        #      加载后直接跳过 warmup（_disc_prewarmed=True）。warmup 完成时由 _save_warmed_disc 写出。
        #   2) resume_ar_path 内的 disc_state_dict —— 同一 run 续训用（_disc_resumed=True）。
        #   3) disc_ckpt —— 前序阶段(如 sf_gan)的 D；序列长度/inst_norm 可能不同，须重新 warmup。
        # 注意：换序列长度 或 翻转 disc_inst_norm 都会改变 D 输入分布；只有 (1) 才跳 warmup，且要确保它确实
        #       是在「相同 frames & inst_norm」下存的（下面会对元数据做一致性 warn）。
        d_sd, src = None, None
        warmed_ckpt = gcfg.get("disc_warmed_ckpt", None)
        if warmed_ckpt and os.path.exists(warmed_ckpt):
            raw = torch.load(warmed_ckpt, map_location="cpu")
            if raw.get("disc_state_dict") is not None:
                d_sd, src = raw["disc_state_dict"], f"disc_warmed_ckpt ({warmed_ckpt})"
                self._disc_prewarmed = True
                if self.is_main():
                    wf, wi = raw.get("frames"), raw.get("disc_inst_norm")
                    if wf is not None and wf != cfg.data.get("frames"):
                        print(f"[D][WARN] warmed D 是在 frames={wf} 上 warmup 的，当前 frames="
                              f"{cfg.data.get('frames')} 不一致 → D 可能仍需重新适应（建议重跑 warmup）。")
                    if wi is not None and wi != self.disc_inst_norm:
                        print(f"[D][WARN] warmed D 的 inst_norm={wi} 与当前 {self.disc_inst_norm} 不一致。")
        resume_ar = cfg.get("resume_ar_path", None)
        if d_sd is None and resume_ar and os.path.exists(resume_ar):
            rck = torch.load(resume_ar, map_location="cpu")
            if rck.get("disc_state_dict") is not None:
                d_sd, src = rck["disc_state_dict"], f"resume_ar_path ({resume_ar})"
                self._disc_resumed = True
        if d_sd is None:
            disc_ckpt = gcfg.get("disc_ckpt", None)
            if disc_ckpt and os.path.exists(disc_ckpt):
                raw = torch.load(disc_ckpt, map_location="cpu")
                d_sd, src = raw.get("disc_state_dict"), f"disc_ckpt ({disc_ckpt})"
        if d_sd is not None:
            miss, unexp = self.disc.load_state_dict(d_sd, strict=False)
            if self.is_main():
                print(f"[D] loaded from {src} (missing={len(miss)}, unexpected={len(unexp)}, "
                      f"resumed={self._disc_resumed}, prewarmed={self._disc_prewarmed})")
        elif self.is_main():
            print("[D] WARNING: no disc weights loaded; D random init.")
        self.disc.train().requires_grad_(True)
        if self.is_main() and self.disc_inst_norm:
            print("[disc_inst_norm=ON] D 输入按 per-sample 标准化；"
                  "请确保 d_warmup_steps 足够 D 在新分布上重新适应。")

    def _setup_deepspeed(self):
        cfg = self.config
        # ---- G engine：DeepSpeed ZeRO ----
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            args=self.args, model=self.model, model_parameters=None,
            config_params=cfg.deepspeed_config_path)

        # ---- D：移出 DeepSpeed，普通 fp32 AdamW（SN parametrization 完好）----
        self.disc_optimizer = None
        if self.gan_enabled:
            self.disc = self.disc.float()
            self.disc_optimizer = torch.optim.AdamW(
                [p for p in self.disc.parameters() if p.requires_grad],
                lr=self.disc_lr, betas=(0.0, 0.999), weight_decay=0.01)
            if self.is_main():
                print(f"[D] outside DeepSpeed | plain AdamW lr={self.disc_lr} | fp32")

        self.start_epoch, self.start_step = self._resume_epoch, self._resume_step
        if self._resume_lr is not None and self.lr_scheduler is not None:
            try:
                self.lr_scheduler.load_state_dict(self._resume_lr)
            except Exception as e:
                if self.is_main():
                    print(f"[Resume] lr_scheduler skip: {e}")

        self.dtype = torch.bfloat16 if self.model_engine.bfloat16_enabled() else torch.float32
        self.denoising_unet.enable_gradient_checkpointing()
        if hasattr(self.motion_predictor, "enable_gradient_checkpointing"):
            self.motion_predictor.enable_gradient_checkpointing()

        # ---- 可选 torch.compile 单帧 AR step（加速 128 帧串行 rollout）----
        # fullgraph=False：遇到 KV cache 变长/对象 mutation 等会 graph-break 回退 eager，不会崩，语义不变。
        # dynamic=True：把增长的 seq_len 当符号维度，避免每步重编译。
        if bool(cfg.get("use_compile", False)):
            global _AR_STEP
            try:
                _AR_STEP = torch.compile(_ar_step_impl, dynamic=True, fullgraph=False)
                if self.is_main():
                    print("[compile] AR step torch.compile(dynamic=True) 已启用（首批 step 会因编译变慢，之后提速）。"
                          "\n          注：KV cache 是增长的，CUDA Graph 抓不住 → 收益有限；大提速需静态 cache（后续做）。")
            except Exception as e:
                if self.is_main():
                    print(f"[compile] 启用失败，回退 eager：{e}")

        if self.gan_enabled and self.is_main():
            import torch.nn.utils.parametrize as P
            n_param = sum(1 for _, m in self.disc.named_modules() if P.is_parametrized(m))
            print(f"[SN check] D parametrized modules: {n_param} "
                  f"({'OK' if n_param > 0 else 'LOST!!!'})")

    def _prepare_data(self):
        cfg = self.config

        def make(src, load_video, ctx_len):
            return MotarDataset(
                pose_dir=src.pose_dir, audio_dir=src.audio_dir, caption_dir=src.caption_dir,
                data_name_path=src.data_name_path, tokenizer_path=cfg.data.tokenizer_path,
                data_stats_path=cfg.data.data_stats_path,
                context_length=ctx_len, fps=cfg.data.fps, sr=cfg.data.sr,
                text_max_len=cfg.data.get("text_max_len", 128),
                random_crop=True, pad_short=(not load_video), load_video=load_video,
                latent_dir=src.latent_dir, video_dir=src.video_dir,
                video_processor=self.video_processor)

        # ---- 正常训练 loader：load_video=True（skip_video 时 False，不解 video/ref，省显存/CPU）----
        main_load_video = not self.skip_video
        ds_main = [make(s, load_video=main_load_video, ctx_len=cfg.data.frames) for s in cfg.data.sources]
        self.train_dataset = ConcatDataset(ds_main) if len(ds_main) > 1 else ds_main[0]
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=cfg.data.batch_size, sampler=self.train_sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=cfg.data.get("prefetch_factor", 3), persistent_workers=True)

        # ---- warmup loader：load_video=False（不解 video/ref，省 CPU）+ 大 batch ----
        self.dwarm_dataloader = None
        if self.gan_enabled and self.d_warmup_steps > 0:
            warm_bs = int(cfg.gan.get("warmup_batch_size", 16))
            warm_workers = int(cfg.gan.get("warmup_num_workers", 4))
            ds_warm = [make(s, load_video=False, ctx_len=cfg.data.frames) for s in cfg.data.sources]
            warm_dataset = ConcatDataset(ds_warm) if len(ds_warm) > 1 else ds_warm[0]
            self.dwarm_sampler = DistributedSampler(warm_dataset, shuffle=True)
            self.dwarm_dataloader = DataLoader(
                warm_dataset, batch_size=warm_bs, sampler=self.dwarm_sampler,
                num_workers=warm_workers, pin_memory=True, drop_last=True,
                prefetch_factor=2, persistent_workers=True)
            if self.is_main():
                print(f"[Dwarm] loader: load_video=False, batch={warm_bs}, workers={warm_workers}")

    def _setup_visualiser(self):
        self.visualiser = None
        if self.skip_video:
            if self.is_main():
                print("[Vis] disabled (skip_video=ON：纯 motion 调试不出视频，避免 UNet OOM)。")
            return
        vis_cfg = self.config.get("visualisation", None)
        if vis_cfg is None or not vis_cfg.get("val_samples"):
            if self.is_main():
                print("[Vis] disabled.")
            return
        from src.visualiser import XNemoARVisualiser
        self.visualiser = XNemoARVisualiser(self.config, self, self.device, self.dtype)

    def _maybe_visualise(self, step):
        vis_interval = self.config.training.get("vis_interval", 0)
        if vis_interval <= 0 or step % vis_interval != 0 or self.visualiser is None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        my_samples = self.visualiser.samples[rank::world]
        if self.is_main():
            print(f"[Vis] @ step {step}")
        self.denoising_unet.disable_gradient_checkpointing()
        try:
            self.visualiser.run(step, self.vis_dir, samples=my_samples)
        except Exception as exc:
            print(f"[Vis] rank {rank} failed @ {step}: {exc}")
        finally:
            self.denoising_unet.enable_gradient_checkpointing()
        if dist.is_initialized():
            dist.barrier()

    def _save_ar_checkpoint(self, epoch, step):
        params = list(self.motion_predictor.parameters())
        stage = 0
        try:
            stage = self.model_engine.zero_optimization_stage()
        except Exception:
            pass

        def _grab(module):
            return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

        if stage == 3:
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                sd = _grab(self.motion_predictor) if self.is_main() else None
        else:
            sd = _grab(self.motion_predictor) if self.is_main() else None
        dd = _grab(self.disc) if (self.gan_enabled and self.is_main()) else None

        if self.is_main():
            ckpt = {"motion_predictor": sd, "step": step, "epoch": epoch}
            if dd is not None:
                ckpt["disc_state_dict"] = dd
            if self.config.training.get("save_lr_state", False) and self.lr_scheduler is not None:
                try:
                    ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
                except Exception:
                    pass
            path = self.ckpt_dir / f"ar_step_{step}.pth"
            torch.save(ckpt, path)
            n = sum(v.numel() for v in sd.values()) / 1e6
            print(f"[Ckpt] saved -> {path} ({n:.1f}M)")
        if dist.is_initialized():
            dist.barrier()

    # ----------------------------- encoding -------------------------------- #
    def _encode_conditions(self, batch, B, T):
        with torch.no_grad():
            text_token = batch["text_token"]
            text_mask = batch.get("text_mask", None)
            if text_mask is not None:
                text_emb = self.text_encoder(input_ids=text_token,
                                             attention_mask=text_mask).last_hidden_state
                text_emb = text_emb * text_mask.unsqueeze(-1)
            else:
                text_emb = self.text_encoder(text_token).last_hidden_state
            wav = normalize_wav(batch["audio_clip"])
            audio_emb = self.audio_encoder(wav).last_hidden_state
            frame_feat = align_to_frames(audio_emb, T)
            local_audio_emb = build_local_window(frame_feat, half=self.window_frames // 2)
            if "ref_img" in batch:
                clip_img_emb = self.image_encoder(
                    batch["ref_img"].to(self.dtype)).image_embeds.unsqueeze(1)
            else:
                clip_img_emb = None
        return text_emb, audio_emb, local_audio_emb, clip_img_emb

    def _window_mask(self, mask_next):
        ws = self.gan_window_size
        Tm1 = mask_next.shape[1]
        Tw = (Tm1 // ws) * ws
        if Tw > 0:
            return mask_next[:, :Tw].reshape(mask_next.shape[0], Tw // ws, ws).any(dim=-1)
        return mask_next[:, :0]

    def _save_warmed_disc(self, global_step):
        """warmup 完成时存下 D，下次设 gan.disc_warmed_ckpt 即可跳过 warmup。
        附带 frames / disc_inst_norm 元数据，供下次加载时做一致性检查。"""
        if not (self.gan_enabled and self.disc is not None):
            return
        if self.is_main():
            dd = {k: v.detach().cpu().clone() for k, v in self.disc.state_dict().items()}
            ckpt = {"disc_state_dict": dd, "step": global_step, "warmed": True,
                    "frames": self.config.data.get("frames"),
                    "disc_inst_norm": self.disc_inst_norm}
            path = self.ckpt_dir / "disc_warmup_done.pth"
            torch.save(ckpt, path)
            print(f"[Dwarm] warmup 完成的 D 已保存 -> {path}\n"
                  f"        下次把 gan.disc_warmed_ckpt 指向它即可跳过 warmup（前提：相同 frames & inst_norm）。")
        if dist.is_initialized():
            dist.barrier()

    def _finish_dwarmup(self, global_step):
        """warmup->正常 切换时统一调用：reset D 的 GAS 计数与残留梯度，记录结束步，并存下 warmup 好的 D。"""
        self._dwarm_done = True
        self._dwarm_done_step = global_step   # lr 退火锚点
        self._d_micro = 0
        if self.disc_optimizer is not None:
            self.disc_optimizer.zero_grad(set_to_none=True)
        self._save_warmed_disc(global_step)   # 持久化 warmup 好的 D
        if self.is_main():
            print(f"[Dwarm DONE] step={global_step} ema_gap={self._ema_d_gap:.3f} "
                  f">= {self.dwarm_gap_target}; D GAS 计数与梯度已 reset；"
                  f"lr 将在 {self.d_post_warmup_anneal} 步内从 {self.d_warmup_lr_boost}x 退回 1x；"
                  f"adv/video now ON")

    def _disc_lr_factor(self, in_d_warmup, global_step):
        """D 当前 lr 倍率：warmup 期=boost；正常期在 anneal 窗口内从 boost 线性退回 1x。"""
        if in_d_warmup:
            return self.d_warmup_lr_boost
        if self.d_post_warmup_anneal <= 0:
            return 1.0
        t = (global_step - self._dwarm_done_step) / max(1, self.d_post_warmup_anneal)
        t = min(1.0, max(0.0, t))
        return self.d_warmup_lr_boost * (1.0 - t) + 1.0 * t

    # ----------------------------- one step -------------------------------- #
    def train_one_step(self, batch, global_step):
        cfg = self.config
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        if "video_tensor" in batch:
            B, T = batch["video_tensor"].shape[:2]
        else:
            B, T = batch["motion_tensor"].shape[:2]

        text_emb, audio_emb, local_audio_emb, clip_img_emb = self._encode_conditions(batch, B, T)

        gt_motion = batch["motion_tensor"]
        frame_mask = batch.get("mask", None)

        in_d_warmup = (self.gan_enabled and (not self._dwarm_done)
                       and (global_step < self.d_warmup_steps))

        lam_video = 0.0 if in_d_warmup else self.lam_video_fn(global_step)
        lam_reg = self.lam_reg_fn(global_step)
        lam_adv = 0.0 if (in_d_warmup or not self.gan_enabled) else self.lam_adv_fn(global_step)

        # 自适应刹车：正常期 D 已被反超（EMA gap < floor）时临时停 adv，让 D 单方面追。
        # 用上一步累积的 _ema_d_gap 判断；不动 video/reg。
        adv_braked = False
        adv_soft_scale = 1.0
        if (not in_d_warmup) and self.gan_enabled and self._dwarm_done:
            braking = (self.adv_gap_floor > 0) and (self._ema_d_gap < self.adv_gap_floor)
            if braking:
                lam_adv = 0.0
                adv_braked = True
            else:
                # 检测「刹车 -> 解除」边沿，记录恢复锚点，启动软启动窗口。
                if self._adv_braked_prev and self.adv_soft_steps > 0:
                    self._adv_resume_step = global_step
                # 软启动窗口内：lam_adv 线性从 0 恢复到满值。
                if self.adv_soft_steps > 0 and self._adv_resume_step >= 0:
                    elapsed = global_step - self._adv_resume_step
                    if elapsed < self.adv_soft_steps:
                        adv_soft_scale = max(0.0, elapsed / float(self.adv_soft_steps))
                        lam_adv *= adv_soft_scale
            self._adv_braked_prev = adv_braked

        local_next = local_audio_emb[:, 1:T]
        mask_next = (frame_mask[:, 1:T] if frame_mask is not None
                     else torch.ones(B, T - 1, device=self.device))

        # 兜底初始化，避免任何分支下 logs 引用未定义变量
        motion_emb = None
        l_video = l_reg = l_adv = l_g = torch.zeros((), device=self.device)

        # ================================================================== #
        # 1) G step
        # ================================================================== #
        if in_d_warmup:
            # warmup：G 只出 fake 给 D，no_grad、不穿 UNet、不更新 G
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
                preds = rollout_lcm(
                    self.motion_predictor, gt_motion, text_emb, audio_emb,
                    local_audio_emb, self.alphas_cp, self.T_max,
                    autocast_factory=lambda: torch.cuda.amp.autocast(dtype=self.dtype))
                seed = gt_motion[:, 0:1].detach()
                motion_emb = torch.cat([seed, preds], dim=1)
        else:
            with torch.enable_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
                # video=0 或 skip_video 时跳过冻结 UNet：只 rollout 出 motion 给 reg/GAN，
                # 省显存/提速 → 可在长 context(如 128) 上调 GAN。
                skip_unet = self.skip_video or (lam_video == 0.0)
                if skip_unet:
                    model_pred, motion_emb = self.model_engine(
                        text_emb, audio_emb, local_audio_emb, None,
                        None, None, None, gt_motion_norm=gt_motion,
                        skip_unet=True, amp_dtype=self.dtype)
                    l_video = torch.zeros((), device=self.device)
                else:
                    if random.random() < cfg.training.get("cfg_drop_prob", 0.1):
                        clip_img_emb = torch.zeros_like(clip_img_emb)

                    latents = batch["video_tensor"].permute(0, 2, 1, 3, 4).contiguous() * self.vae_scale
                    if torch.isnan(latents).any():
                        return None, None, "Latents NaN"
                    ref_image_latents = batch["ref_latent"] * self.vae_scale

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.num_train_timesteps, (B,),
                                              device=self.device).long()
                    noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)

                    model_pred, motion_emb = self.model_engine(
                        text_emb, audio_emb, local_audio_emb, clip_img_emb,
                        ref_image_latents, noisy_latents, timesteps, gt_motion_norm=gt_motion,
                        skip_unet=False, amp_dtype=noisy_latents.dtype)

                    if self.pred_type == "epsilon":
                        target = noise
                    elif self.pred_type == "v_prediction":
                        target = self.train_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(self.pred_type)

                    lat_h, lat_w = latents.shape[-2], latents.shape[-1]
                    weight_map = build_mouth_weight_map(
                        cfg.loss.get("mouth_mode", "region"), lat_h, lat_w,
                        cfg.loss.get("mouth_gain", 3.0), self.device,
                        masks=batch.get("mouth_mask", None),
                        region=tuple(cfg.loss.get("mouth_region", [0.55, 0.92, 0.28, 0.72])))

                    l_video = ddpm_video_loss(model_pred, target, weight_map=weight_map,
                                              frame_mask=frame_mask, frame_dim=2)

                l_reg = motion_regression_loss(motion_emb, gt_motion, frame_mask=frame_mask,
                                               kind=cfg.loss.get("reg_kind", "smooth_l1"))

                # ---- adv：G 借 D 算梯度。----
                # [改] 不再 disc.eval()：保持 disc 在 train 模式（与 S4 一致），让 G 对抗的是
                #   「正在被训练的那个 D」（SN 幂迭代/cond_drop 与 D-step 同步）。若切 eval，G 会去钻
                #   eval-态 D（缓存 SN、无 cond_drop）的空子，而 D-step 训的是 train-态 D，二者不一致 → 失稳翻转。
                #   仍保留 requires_grad_(False)：S5 的 gas=8 不每步 zero_grad，需冻结防止 G-backward 污染 D 累积梯度。
                l_adv = torch.zeros((), device=self.device)
                if self.gan_enabled and lam_adv > 0:
                    preds_motion = self._disc_in(motion_emb[:, 1:])
                    for p in self.disc.parameters():
                        p.requires_grad_(False)
                    with torch.cuda.amp.autocast(enabled=False):
                        f_fake, w_fake = self.disc(
                            preds_motion.float(), text_emb.float(),
                            audio_emb.float(), local_next.float())
                    mask_win = self._window_mask(mask_next)
                    l_g_frame = ns_g_loss(f_fake.float(), mask_next)
                    l_g_win = (ns_g_loss(w_fake.float(), mask_win) if mask_win.numel()
                               else torch.zeros((), device=self.device))
                    l_adv = l_g_frame + self.w_win * l_g_win

                # adv_grad_off：副作用全保留（D forward 已执行），但梯度清零
                lam_adv_eff = 0.0 if self.adv_grad_off else lam_adv
                l_g = lam_video * l_video + lam_reg * l_reg + lam_adv_eff * l_adv

            if torch.isnan(l_g):
                return None, None, "Loss NaN"

            self.model_engine.backward(l_g)
            self.model_engine.step()

        # ================================================================== #
        # 2) D step（普通 AdamW）
        #    warmup 期每步 step；正常期按 GAS 累积（对齐 G，1:1）
        # ================================================================== #
        d_real_v = d_fake_v = float("nan")
        l_d_v = float("nan")
        d_r1_v = float("nan")
        if self.gan_enabled:
            for p in self.disc.parameters():
                p.requires_grad_(True)
            self.disc.train()

            cur_lr = self.disc_lr * self._disc_lr_factor(in_d_warmup, global_step)
            for pg in self.disc_optimizer.param_groups:
                pg["lr"] = cur_lr

            # real / fake 进 D 前统一实例标准化
            preds_for_d = self._disc_in(motion_emb[:, 1:].detach())
            gt_next = self._disc_in(gt_motion[:, 1:T])
            mask_win = self._window_mask(mask_next)

            # D:G 更新比例。正常期把 D 的有效 GAS 缩小为 gas//d_ratio，
            # 等效 D 在同样 G-step 间隔内多更新 d_ratio 倍。warmup 期仍每步 step。
            gas = 1 if in_d_warmup else max(1, self.gas // self.d_ratio)
            if self._d_micro % gas == 0:
                self.disc_optimizer.zero_grad(set_to_none=True)

            # ---- R1（可选，默认关）：对已标准化的 gt_next 求 ----
            if (self.r1_gamma > 0) and (not in_d_warmup) and (self._d_micro % self.r1_every == 0):
                real_m = gt_next.detach().float().clone().requires_grad_(True)
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    f_r1, _ = self.disc(real_m, text_emb.float(),
                                        audio_emb.float(), local_next.float())
                m_r1 = mask_next.unsqueeze(-1).float()
                out = (f_r1.float() * m_r1).sum()
                (grad,) = torch.autograd.grad(out, real_m, create_graph=True)
                r1 = grad.pow(2).flatten(1).sum(1).mean() * 0.5
                (self.r1_gamma * r1 / gas).backward()
                d_r1_v = r1.detach().item()

            with torch.cuda.amp.autocast(enabled=False):
                motion_cat = torch.cat([gt_next, preds_for_d], dim=0).float()
                text_cat = torch.cat([text_emb, text_emb], dim=0).float()
                audio_cat = torch.cat([audio_emb, audio_emb], dim=0).float()
                local_cat = torch.cat([local_next, local_next], dim=0).float()
                f_cat, w_cat = self.disc(motion_cat, text_cat, audio_cat, local_cat)
                bsz = gt_next.shape[0]
                f_real, f_fake_d = f_cat[:bsz], f_cat[bsz:]
                w_real, w_fake_d = w_cat[:bsz], w_cat[bsz:]
                l_d_frame = ns_d_loss(f_real.float(), f_fake_d.float(), mask_next)
                l_d_win = (ns_d_loss(w_real.float(), w_fake_d.float(), mask_win)
                           if mask_win.numel() else torch.zeros((), device=self.device))
                l_d_raw = l_d_frame + self.w_win * l_d_win
                l_d = l_d_raw / gas

            l_d.backward()

            self._d_micro += 1
            if self._d_micro % gas == 0:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    world = dist.get_world_size()
                    for p in self.disc.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world)
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.d_grad_clip)
                self.disc_optimizer.step()

            d_real_v = f_real.detach().float().mean().item()
            d_fake_v = f_fake_d.detach().float().mean().item()
            l_d_v = l_d_raw.detach().item()

            if self._dbg(global_step) and not in_d_warmup:
                r1s = d_r1_v if d_r1_v == d_r1_v else 0.0
                soft_tag = (f" soft={adv_soft_scale:.2f}"
                            if (self.adv_soft_steps > 0 and adv_soft_scale < 1.0) else "")
                print(f"[Dstep] gs={global_step} f_real={d_real_v:+.3f} "
                      f"f_fake={d_fake_v:+.3f} gap={d_real_v - d_fake_v:+.3f} "
                      f"ema_gap={self._ema_d_gap:+.3f} l_d={l_d_v:.3f} r1={r1s:.3f} "
                      f"lr={cur_lr:.2e} d:g={self.d_ratio}:1 "
                      f"{'[ADV-BRAKED]' if adv_braked else ''}{soft_tag}")

            if in_d_warmup:
                gap = d_real_v - d_fake_v
                self._ema_d_gap = (self.dwarm_ema_alpha * gap
                                   + (1 - self.dwarm_ema_alpha) * self._ema_d_gap)
                flip = ((global_step >= self.dwarm_min_steps)
                        and (self._ema_d_gap >= self.dwarm_gap_target))
                if dist.is_initialized() and dist.get_world_size() > 1:
                    flip_t = torch.tensor([1 if flip else 0], device=self.device)
                    dist.all_reduce(flip_t, op=dist.ReduceOp.MIN)
                    flip = bool(flip_t.item())
                if flip and not self._dwarm_done:
                    self._finish_dwarmup(global_step)
            else:
                # 正常期也持续维护 EMA gap，供 adv 自适应刹车判断。
                # [修] 多卡下跨 rank 同步 gap（mean），否则各 rank 的 _ema_d_gap 独立演化，
                # 刹车判断 (gap < adv_gap_floor) 可能一卡刹车一卡不刹 → λ_adv 跨 rank 不一致 →
                # DeepSpeed all-reduce 出来的是两个不同加权损失的梯度。
                gap = d_real_v - d_fake_v
                if dist.is_initialized() and dist.get_world_size() > 1:
                    gap_t = torch.tensor([gap], device=self.device)
                    dist.all_reduce(gap_t, op=dist.ReduceOp.SUM)
                    gap = gap_t.item() / dist.get_world_size()
                self._ema_d_gap = (self.dwarm_ema_alpha * gap
                                   + (1 - self.dwarm_ema_alpha) * self._ema_d_gap)

        logs = {
            "loss": l_g.item() if torch.is_tensor(l_g) else 0.0,
            "l_video": l_video.item() if torch.is_tensor(l_video) else 0.0,
            "l_reg": l_reg.item() if torch.is_tensor(l_reg) else 0.0,
            "l_adv": float(l_adv.item()) if torch.is_tensor(l_adv) else 0.0,
            "lam_video": lam_video, "lam_reg": lam_reg, "lam_adv": lam_adv,
            "adv_braked": 1.0 if adv_braked else 0.0,
            "adv_soft": adv_soft_scale,
            "ema_gap": self._ema_d_gap,
            "std_ratio": std_ratio(motion_emb.detach(), gt_motion) if motion_emb is not None else float("nan"),
            "d_real": d_real_v, "d_fake": d_fake_v, "l_d": l_d_v, "d_r1": d_r1_v,
            "dwarm": 1.0 if in_d_warmup else 0.0,
        }
        return logs, None, None

    # ----------------------------- loop ------------------------------------ #
    def _run_error_synced(self, batch, global_step):
        """跑一步并跨 rank 同步 error。返回 (logs or None, errored: bool)。"""
        local_error = 0
        logs = None
        try:
            logs, _, err = self.train_one_step(batch, global_step)
            if err is not None:
                local_error = 1
                if self.is_main():
                    print(f"Warn: {err} @ {global_step}, skip")
        except Exception as e:
            local_error = 1
            print(f"Rank {dist.get_rank()} Exception @ {global_step}: {e}")
        err_t = torch.tensor([local_error], device=self.device)
        if dist.is_initialized():
            dist.all_reduce(err_t, op=dist.ReduceOp.SUM)
        if err_t.item() > 0:
            try:
                self.ref_reader.clear()
                self.ref_writer.clear()
            except Exception:
                pass
            return None, True
        return logs, False

    def _mark_dwarmup_skipped(self, global_step):
        """resume 越过 warmup 时统一标记完成：置 _dwarm_done、退火锚点前移使 D lr 倍率立即=1x、
        reset D 的 GAS 计数与残留梯度。否则 _dwarm_done 恒 False，正常期 adv 自适应刹车
        (adv_gap_floor) + 软启动 (adv_soft_steps) + lr 退火 会被静默关闭。"""
        self._dwarm_done = True
        self._dwarm_done_step = global_step - self.d_post_warmup_anneal
        self._d_micro = 0
        if self.disc_optimizer is not None:
            self.disc_optimizer.zero_grad(set_to_none=True)

    def _run_dwarmup_phase(self, global_step):
        # D 已从 disc_warmed_ckpt 预热好：直接跳过 warmup（不论 global_step，含 fresh run 从 0 起）。
        if self.gan_enabled and not self._dwarm_done and self._disc_prewarmed:
            self._mark_dwarmup_skipped(global_step)
            if self.is_main():
                print(f"[Phase0] D 已从 disc_warmed_ckpt 预热 → 跳过 warmup @ step {global_step}；"
                      f"D lr=1x，adv 刹车/软启动已启用。")
            return global_step
        # resume 已越过 warmup 预算（global_step >= d_warmup_steps，基于 global_step 的 warmup 跑不起来）。
        if self.gan_enabled and not self._dwarm_done and global_step >= self.d_warmup_steps:
            self._mark_dwarmup_skipped(global_step)
            if self.is_main():
                if self._disc_resumed:
                    # D 来自同一 run 的 ckpt，已适配当前 (序列长度, inst_norm)：跳过 warmup 是安全的。
                    print(f"[Phase0] D resumed & past warmup budget @ {global_step}: "
                          f"skip warmup, mark done, D lr=1x, adv 刹车/软启动已启用。")
                else:
                    # D 来自 disc_ckpt（前序阶段），序列长度/inst_norm 很可能不同 → D 未适配当前分布，
                    # 而 global_step 又超了 warmup 预算 → 这是配置问题。最干净的修法是 resume_ar_path=null
                    # 走一次从头的 warmup（global_step 从 0 起）。
                    print(f"[Phase0][WARN] D 未从本 run resume，且 global_step({global_step}) >= "
                          f"d_warmup_steps({self.d_warmup_steps})：warmup 跑不起来，D 可能未适配当前"
                          f"(序列长度/inst_norm)分布。强烈建议 resume_ar_path=null 做一次干净 warmup。"
                          f"暂标记 dwarm 完成以启用 braking。")
            return global_step
        if not (self.gan_enabled and self.dwarm_dataloader is not None
                and not self._dwarm_done and global_step < self.d_warmup_steps):
            return global_step
        if self.is_main():
            print(f"[Phase0] D-warmup start (max {self.d_warmup_steps} steps)")
        warm_epoch = 0
        while (not self._dwarm_done) and global_step < self.d_warmup_steps:
            self.dwarm_sampler.set_epoch(warm_epoch)
            for batch in self.dwarm_dataloader:
                if self._dwarm_done or global_step >= self.d_warmup_steps:
                    break
                logs, errored = self._run_error_synced(batch, global_step)
                if errored:
                    continue
                global_step += 1
                if self.is_main() and logs is not None and global_step % self.log_every == 0:
                    print(f"[Dwarm step{global_step}] d_real={logs['d_real']:+.3f} "
                          f"d_fake={logs['d_fake']:+.3f} gap={logs['d_real'] - logs['d_fake']:+.3f} "
                          f"ema_gap={self._ema_d_gap:+.3f} l_d={logs['l_d']:.4f}")
            warm_epoch += 1
        # 若因步数上限退出（而非 ema 触发），同样走统一收尾保证 reset
        if not self._dwarm_done:
            self._finish_dwarmup(global_step)
        if self.is_main():
            print(f"[Phase0] D-warmup done @ step {global_step} (ema_gap={self._ema_d_gap:+.3f})")
        return global_step

    def fit(self):
        cfg = self.config
        global_step = self.start_step
        run_logs = {}

        # ---- Phase 0: D-warmup ----
        global_step = self._run_dwarmup_phase(global_step)

        # ---- Phase 1+: 正常 video + adv ----
        for epoch in range(self.start_epoch, cfg.training.num_epochs):
            self.train_sampler.set_epoch(epoch)
            for step, batch in enumerate(self.train_dataloader):
                logs, errored = self._run_error_synced(batch, global_step)
                if errored:
                    continue
                global_step += 1

                if self.is_main() and logs is not None:
                    for k, v in logs.items():
                        if v == v:  # 过滤 NaN
                            run_logs[k] = run_logs.get(k, 0.0) + v
                            run_logs[k + "_n"] = run_logs.get(k + "_n", 0) + 1
                    for k in ("loss", "l_video", "l_reg", "l_adv", "std_ratio",
                              "d_real", "d_fake", "l_d", "lam_video", "lam_reg", "lam_adv"):
                        self.writer.add_scalar(f"train/{k}", logs.get(k, 0.0), global_step)
                    if global_step % self.log_every == 0:
                        def avg(k):
                            n = run_logs.get(k + "_n", 0)
                            return run_logs.get(k, 0.0) / n if n > 0 else float("nan")

                        msg = (f"loss={avg('loss'):.4f} l_video={avg('l_video'):.4f} "
                               f"l_reg={avg('l_reg'):.4f} l_adv={avg('l_adv'):.4f} "
                               f"std={avg('std_ratio'):.4f}")
                        if self.gan_enabled:
                            msg += (f" | D(real={avg('d_real'):+.3f} fake={avg('d_fake'):+.3f} "
                                    f"gap={avg('d_real') - avg('d_fake'):+.3f})")
                        print(f"[{datetime.datetime.now()}] ep{epoch} step{global_step} {msg} "
                              f"lam_v={logs['lam_video']:.3f} lam_r={logs['lam_reg']:.3f} "
                              f"lam_a={logs['lam_adv']:.3f}")
                        run_logs = {}

                self._maybe_visualise(global_step)
                if global_step % cfg.training.save_interval == 0:
                    self._save_ar_checkpoint(epoch, global_step)


# =========================================================================== #
# config 建议（GAN + video 联合优化）：
#   loss:
#     lambda_video: { start_val: 1.0, end_val: 0.15, warmup_start: <warmup结束步>, warmup_end: <+几千步> }
#     lambda_reg:   0.1
#     lambda_adv:   { start_val: 0.0, end_val: 0.05, warmup_start: <warmup结束步>, warmup_end: <+5000> }
#   gan:
#     disc_inst_norm: true        # 掐掉方差捷径；开启后 D 必须重新 warmup
#     d_warmup_steps: 1500        # 给足 D 在标准化输入分布上重新适应
# 说明：adv ramp 区间应与 video 降权区间重叠/对齐——adv 上来时 video 退下去，
#       二者不再正交对抗，GAN 才稳。
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    deepspeed.init_distributed()
    config = OmegaConf.load(args.config)
    XNemoARTrainer(config, args).fit()
"""
verify_ar_xnemo_sweep_mp.py
===========================
verify_ar_xnemo_sweep.py 的多卡数据并行版本。

并行方式:
    - 进程级数据并行: 每张 GPU 一个独立子进程, 各自完整加载编码器 / X-Nemo
      pipeline / AR 模型 (推理无梯度同步)。
    - 切分粒度: 样本 (sample) 维度。每个 worker 处理 samples[rank::world_size]。
      在每个样本内部, 仍按原逻辑串行扫一遍 ar_sampling_steps (因为 GT 视频
      要在该样本内复用、AR 模型每步数都要重建)。
    - 报告: 每个 worker 写出 .partial_report_rank{rank}.json, 主进程在所有
      worker 退出后合并成 report.json 并打印汇总。

使用:
    python verify_ar_xnemo_sweep_mp.py \
        --config ... --ar_ckpt ... --test_dir ... \
        --gpus 0,1,2,3 \
        --ar_sampling_steps 5,10,20,50 ...

参数差异 (相对单卡版本):
    --device  -> --gpus     接收逗号分隔的 GPU id, e.g. "0,1,2,3"

其它逻辑 (GT 对照、seed_mode、CFG、context 窗口、metrics) 保持不变。
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# 路径: 把 AR 工程根目录 与 x-nemo 工程根目录 加入 sys.path
# ---------------------------------------------------------------------------
# TODO: 改成你的实际路径
AR_REPO_ROOT     = os.environ.get("AR_REPO_ROOT",     "/path/to/ar_repo")
XNEMO_REPO_ROOT  = os.environ.get("XNEMO_REPO_ROOT",  "/media/ps/ssd5/ayr/x-nemo-inference")
for p in (AR_REPO_ROOT, XNEMO_REPO_ROOT):
    if p and p not in sys.path:
        sys.path.append(p)

# ---- 编码器 / AR 模型 ----
import librosa
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from transformers import (
    AutoTokenizer, UMT5EncoderModel, Wav2Vec2Model, Wav2Vec2Processor,
    CLIPVisionModelWithProjection,
)

from model.armodel import MotionTransformer
from utils.processor import get_windowed_audio

# ---- X-Nemo 解码器组件 ----
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
from src.pipelines.pipeline_pose2vid_motenc_long import (
    Pose2VideoPipeline, Pose2VideoPipelineOutput,
)
from src.pipelines.context import get_context_scheduler
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.utils.util import save_videos_grid

import subprocess
import shutil

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CHUNK_FRAMES    = 128
AUDIO_SR        = 16000
WINDOW_SIZE     = 5
FPS             = 25
BBOX_PARAM_DIM  = 3
MOTION_TOKEN_SHAPE = (32, 16)


def normalize_wav(wav):  # [B, N]
    return (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-7)

def align_to_frames(feat, T):  # [B, A, C] -> [B, T, C]
    feat = feat.transpose(1, 2)
    feat = F.interpolate(feat.float(), size=T, mode="linear", align_corners=False)
    return feat.transpose(1, 2)

def build_local_window(frame_feat, half=2):  # [B,T,C] -> [B,T,2h+1,C]
    feat = frame_feat.transpose(1, 2)
    feat = F.pad(feat, (half, half), mode="replicate")
    windows = feat.unfold(-1, 2 * half + 1, 1)
    return windows.permute(0, 2, 3, 1).contiguous()

# ===========================================================================
#  注入版 Pipeline (与原脚本完全一致)
# ===========================================================================
class ARMotionPose2VideoPipeline(Pose2VideoPipeline):

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        ref_pose_image,
        motion_latents,            # [1, T, motion_dim]
        width, height, video_length,
        num_inference_steps, guidance_scale,
        generator=None, init_latents=None,
        context_schedule="uniform", context_frames=24, context_stride=1,
        context_overlap=4, context_batch_size=1,
        eta: float = 0.0, output_type="tensor", return_dict=True,
        **kwargs,
    ):
        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_size = 1

        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        if do_cfg:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet, do_classifier_free_guidance=do_cfg,
            mode="write", batch_size=batch_size, fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet, do_classifier_free_guidance=do_cfg,
            mode="read", batch_size=batch_size, fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size, num_channels_latents, width, height, video_length,
            clip_image_embeds.dtype, device, generator, init_latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        ).to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean * 0.18215

        repeated_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        repeated_latents = self.downgrade_input(
            repeated_latents, generator, device, ref_image_latents.dtype
        )
        # 用 seeded generator（独立于被 AR 生成污染的全局 RNG）→ 解码器初始噪声跨模型完全一致，
        # 这样不同模型同一样本的视频差异 = 纯 motion 差异，可公平对比。
        noise = torch.randn(
            repeated_latents.shape, generator=generator,
            device=repeated_latents.device, dtype=repeated_latents.dtype,
        )
        noisy_first = []
        for fidx in range(video_length):
            noisy_first.append(self.scheduler.add_noise(
                repeated_latents[:, :, fidx:fidx + 1],
                noise[:, :, fidx:fidx + 1], timesteps[:1]))
        latents = torch.cat(noisy_first, dim=2)

        context_scheduler = get_context_scheduler(context_schedule)
        motion_latents = motion_latents.to(device=device, dtype=self.motion_encoder.dtype)

        neg_motion_hidden_states = None
        if do_cfg:
            ref_mot_bbox_param = torch.ones((1, BBOX_PARAM_DIM),
                                            device=device, dtype=self.motion_encoder.dtype)
            ref_mot_bbox_param[:, :2] *= 0
            ref_pose_cond_tensor = self.cond_image_processor.preprocess(
                ref_pose_image, height=224, width=224
            ).to(device=device, dtype=self.motion_encoder.dtype)
            neg_motion_hidden_states = self.motion_encoder(
                ref_pose_cond_tensor, ref_mot_bbox_param)

        if neg_motion_hidden_states is not None:
            token_shape = tuple(neg_motion_hidden_states.shape[1:])
        else:
            token_shape = MOTION_TOKEN_SHAPE
        per_frame = int(np.prod(token_shape))

        def select_motion(frames: List[int]) -> torch.Tensor:
            win = motion_latents[:, frames]
            assert win.shape[-1] == per_frame, (
                f"AR 每帧 motion 维度 {win.shape[-1]} 与 encoder 的 {token_shape}"
                f"(={per_frame}) 不一致, 请检查 motion_dim")
            return win.reshape(1, win.shape[1], *token_shape)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, t in enumerate(timesteps):
                if step_idx == 0:
                    self.reference_unet(
                        ref_image_latents.repeat((2 if do_cfg else 1), 1, 1, 1),
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    # ★ 2026-07-26: update() 的 dtype 参数默认硬编码 fp16, 跑 bf16(本文件/bench脚本
                    # 统一用的 dtype)时这里不传会在后面 reference attention 里炸
                    # "expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16"
                    # (bank 张量固化成 fp16/fp32、跟 bf16 的 denoising_unet 权重对不上)。同一个坑在
                    # AniPortrait/src/pipelines/pipeline_pose2vid_long.py 和 x-nemo-inference/src/
                    # pipelines/causal_streaming_pipeline.py 的同名调用点已经修过, 这里补上。
                    reference_control_reader.update(reference_control_writer, dtype=self.denoising_unet.dtype)

                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_cfg else 1), *latents.shape[1:]),
                    device=latents.device, dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device, dtype=latents.dtype,
                )

                offset = random.randint(0, context_frames - 1)
                context_queue = list(context_scheduler(
                    0, num_inference_steps, latents.shape[2],
                    context_frames, context_stride, context_overlap, True, offset,
                ))
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = [
                    context_queue[ci * context_batch_size:(ci + 1) * context_batch_size]
                    for ci in range(num_context_batches)
                ]

                for context in global_context:
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context]).to(device)
                        .repeat(2 if do_cfg else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    motion_hidden_states = torch.cat(
                        [select_motion(c) for c in context], dim=0)
                    if do_cfg:
                        motion_hidden_states = torch.cat(
                            [neg_motion_hidden_states.unsqueeze(1).expand_as(motion_hidden_states),
                             motion_hidden_states], dim=0)

                    pred = self.denoising_unet(
                        latent_model_input, t,
                        encoder_hidden_states=[image_prompt_embeds, motion_hidden_states],
                        pose_cond_fea=None, return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                noise_pred = noise_pred / counter
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if step_idx == len(timesteps) - 1 or (
                    (step_idx + 1) > num_warmup_steps
                    and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

            reference_control_reader.clear()
            reference_control_writer.clear()

        if isinstance(self.vae, AutoencoderKLTemporalDecoder):
            images = self.decode_latents_svd(latents)
        else:
            images = self.decode_latents(latents)
        if output_type == "tensor":
            images = torch.from_numpy(images)
        if not return_dict:
            return images
        return Pose2VideoPipelineOutput(videos=images)


# ===========================================================================
#  模型加载
# ===========================================================================
def load_encoders(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_path)
    text_encoder = UMT5EncoderModel.from_pretrained(
        config.text_encoder_path).to(device, dtype=torch.float32).eval()
    audio_processor = Wav2Vec2Processor.from_pretrained(config.audio_encoder_path)
    audio_encoder = Wav2Vec2Model.from_pretrained(
        config.audio_encoder_path).to(device, dtype=torch.float32).eval()
    return tokenizer, text_encoder, audio_processor, audio_encoder


def load_ar_model(config, ckpt, device, dtype, num_sampling_steps):
    """每次以不同的 num_sampling_steps 实例化 AR 模型。
    步数写入 diffloss 采样器, 必须在构造时指定, 不能事后修改。"""
    ar_model = MotionTransformer(
        motion_dim=config.get("motion_dim", 512),
        depth=config.get("depth", 12),
        heads=config.get("heads", 8),
        dim_head=config.get("dim_head", 64),
        diffloss_dim=config.get("diffloss_dim", 512),
        diffloss_depth=config.get("diffloss_depth", 4),
        num_sampling_steps=num_sampling_steps,
        pretrained_path=ckpt,
    ).to(device, dtype=dtype).eval()
    return ar_model


def load_xnemo_pipeline(config, device, weight_dtype):
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config.vae_path, weight_dtype=weight_dtype).to(device, dtype=weight_dtype)
    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path).to(dtype=weight_dtype, device=device)
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(
        torch.load(config.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                   map_location="cpu"), strict=True)
    motion_encoder.load_state_dict(
        torch.load(config.denoising_unet_path.replace("denoising_unet", "motion_encoder"),
                   map_location="cpu"), strict=True)
    denoising_unet.load_state_dict(
        torch.load(config.temporal_module_path, map_location="cpu"), strict=False)

    pipe = ARMotionPose2VideoPipeline(
        vae=vae, image_encoder=image_enc, reference_unet=reference_unet,
        denoising_unet=denoising_unet, motion_encoder=motion_encoder, scheduler=scheduler,
    ).to(device, dtype=weight_dtype)
    return pipe


# ===========================================================================
#  特征提取
# ===========================================================================
@torch.no_grad()
def extract_features(sample, tokenizer, text_encoder, audio_processor, audio_encoder, device):
    # ---- audio: 整段 ----
    audio, _ = librosa.load(sample["audio_path"], sr=AUDIO_SR)
    total_frames = len(audio) // (AUDIO_SR // FPS)        # = T
    audio = audio[: total_frames * (AUDIO_SR // FPS)]
    audio_clip = torch.from_numpy(audio).float().unsqueeze(0).to(device)   # [1, T*640]

    wav = normalize_wav(audio_clip)
    audio_emb = audio_encoder(wav).last_hidden_state       # [1,A,C]
    frame_feat = align_to_frames(audio_emb, total_frames)  # [1,T,C]
    local_audio_emb = build_local_window(frame_feat, half=WINDOW_SIZE // 2)  # [1,T,5,C]

    # ---- text ----
    caption = load_caption(sample["caption_path"])
    tok = tokenizer(caption, padding="max_length", truncation=True,
                    max_length=128, return_tensors="pt")
    tokens = tok.input_ids.to(device)
    tmask  = tok.attention_mask.to(device)
    with torch.cuda.amp.autocast(dtype=torch.float32):
        text_emb = text_encoder(input_ids=tokens, attention_mask=tmask).last_hidden_state
    text_emb = text_emb * tmask.unsqueeze(-1)

    return {
        "audio_emb": audio_emb,
        "local_audio_emb": local_audio_emb,
        "text_emb": text_emb,
        "total_frames": total_frames,
    }


def load_caption(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return list(data.values())[0] if isinstance(data, dict) else data[0]
    except Exception:
        return "A person speaking"


def load_motion_gt(path):
    emb = torch.load(path, map_location="cpu")
    if emb.dim() == 4:
        emb = emb.squeeze(0).reshape(emb.shape[1], -1)
    elif emb.dim() == 3:
        emb = emb.squeeze(0)
    return emb.float()


# ===========================================================================
#  AR motion 生成
# ===========================================================================
@torch.no_grad()
def compute_ref_motion_emb(pipe, ref_pose_pil, device, dtype):
    ref_bbox_param = torch.ones((1, BBOX_PARAM_DIM), device=device,
                                dtype=pipe.motion_encoder.dtype)
    ref_bbox_param[:, :2] *= 0
    ref_cond = pipe.cond_image_processor.preprocess(
        ref_pose_pil, height=224, width=224
    ).to(device=device, dtype=pipe.motion_encoder.dtype)
    emb = pipe.motion_encoder(ref_cond, ref_bbox_param)
    return emb.reshape(1, 1, -1).to(dtype)


@torch.no_grad()
def generate_ar_motion(features, ar_model, device, dtype,
                       cfg_audio=4.0, cfg_text=2.0, seed_frame=None, use_lcm=False,
                       temperature=1.0, drop_global=False, drop_local=False):
    T = features["total_frames"]
    text_emb        = features["text_emb"].to(device, dtype=dtype)
    audio_emb       = features["audio_emb"].to(device, dtype=dtype)
    local_audio_emb = features["local_audio_emb"].to(device, dtype=dtype)
    seed = seed_frame.to(device, dtype=dtype) if seed_frame is not None else None

    # use_lcm 时：lcm_steps = 采样步数（>1 走 sample_for_infer 多步 DDIM，= 蒸馏标定的 4 步采样器）。
    # 关键：单走 num_sampling_steps 对 LCM 路径无效——sample_lcm_dual_cfg 永远 1 步、无视步数。
    n_steps = ar_model.diffloss.num_sampling_steps
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = ar_model.generate(
            seq_len=T, text_emb=text_emb,
            audio_emb=audio_emb,
            local_audio_feat=local_audio_emb,
            cfg_audio=cfg_audio, cfg_text=cfg_text,
            use_tqdm=False, first_frame=seed,
            num_sampling_steps=n_steps,
            denorm_output=True,
            use_lcm=use_lcm,
            lcm_steps=(n_steps if use_lcm else 1),
            temperature=temperature,
            # condition 消融：置零点与训练期 cond_drop 一致（fusion_net 后 / audio_proj 后）
            drop_global=drop_global, drop_local=drop_local,
        )
    return pred.squeeze(0)


# ===========================================================================
#  度量
# ===========================================================================
def compute_metrics(pred, gt):
    T = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:T].float(), gt[:T].float()
    return {
        "frames":     T,
        "mse":        F.mse_loss(pred, gt).item(),
        "mae":        (pred - gt).abs().mean().item(),
        "cos_sim":    F.cosine_similarity(pred, gt, dim=-1).mean().item(),
        "std_ratio":  (pred.std(0).mean() / (gt.std(0).mean() + 1e-8)).item(),
    }


# ===========================================================================
#  渲染
# ===========================================================================
@torch.no_grad()
def render(pipe, ref_image_pil, ref_pose_pil, motion_latents, args, generator):
    T = motion_latents.shape[1]
    out = pipe(
        ref_image=ref_image_pil,
        ref_pose_image=ref_pose_pil,
        motion_latents=motion_latents,
        width=args.W, height=args.H, video_length=T,
        num_inference_steps=args.steps, guidance_scale=args.cfg,
        generator=generator,
        context_frames=args.context_frames, context_overlap=args.context_overlap,
    )
    return out.videos


# ===========================================================================
#  数据集 IO
# ===========================================================================
def collect_samples(test_dir, max_samples=None):
    folders = sorted([f for f in os.listdir(test_dir) if f.isdigit()],
                     key=lambda x: int(x))
    if max_samples is not None:
        folders = folders[:max_samples]
    samples = []
    for folder in folders:
        base      = os.path.join(test_dir, folder)
        audio_dir = os.path.join(base, "audio_wav")
        pose_dir  = os.path.join(base, "pose_embed")
        frames_dir= os.path.join(base, "face_frames")
        if not os.path.isdir(audio_dir):
            continue
        wavs = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        if not wavs:
            continue
        sid   = os.path.splitext(wavs[0])[0]
        poses = [f for f in os.listdir(pose_dir) if f.endswith(".pt")] \
                if os.path.isdir(pose_dir) else []
        frames= sorted(os.listdir(frames_dir)) if os.path.isdir(frames_dir) else []
        samples.append({
            "name":        folder,
            "audio_path":  os.path.join(audio_dir, wavs[0]),
            "pose_path":   os.path.join(pose_dir, poses[0]) if poses else None,
            "caption_path":os.path.join(base, "emo_pose_caption", f"{sid}.json"),
            "ref_frame":   os.path.join(frames_dir, frames[0]) if frames else None,
        })
    return samples


def mux_audio_to_video(video_path: str, audio_path: str, fps: int = FPS) -> None:
    if not shutil.which("ffmpeg"):
        print("  [warn] ffmpeg 未找到, 跳过音频合并")
        return
    if not os.path.exists(audio_path):
        print(f"  [warn] 音频文件不存在: {audio_path}, 跳过音频合并")
        return
    tmp_path = video_path.replace(".mp4", "_with_audio.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] ffmpeg 合并失败:\n{result.stderr}")
        return
    os.replace(tmp_path, video_path)


# ===========================================================================
#  参数解析
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    type=str, required=True)
    ap.add_argument("--ar_ckpt",   type=str, required=True)
    ap.add_argument("--test_dir",  type=str, required=True)
    ap.add_argument("--ref_image", type=str, default=None)
    ap.add_argument("--output_dir",type=str, default="output/verify_ar_xnemo_sweep")
    # ----------------------------------------------------------------
    # 多卡: 用 --gpus 指定可用 GPU id 列表, 每张卡起一个子进程
    # ----------------------------------------------------------------
    ap.add_argument("--gpus", type=str, default="0",
                    help="逗号分隔的 GPU id 列表, 如 '0,1,2,3'。每张卡一个进程")
    ap.add_argument("--num_samples",type=int, default=5)
    ap.add_argument("--sample_ids", type=str, default="",
                    help="逗号分隔的 test 文件夹名(如 '2,4')，只渲指定样本；设了则忽略 --num_samples")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("-W",          type=int, default=512)
    ap.add_argument("-H",          type=int, default=512)
    ap.add_argument("--steps",     type=int, default=35, help="视频扩散步数 (X-Nemo 解码器)")
    ap.add_argument("--cfg",       type=float, default=2.5)
    ap.add_argument(
        "--ar_sampling_steps",
        type=str,
        default="5,10,20,50",
        help="逗号分隔的多个 AR diffloss 采样步数, 如 '5,10,20,50'",
    )
    ap.add_argument("--ar_cfg_audio", type=float, default=4.0)
    ap.add_argument("--ar_cfg_text",  type=float, default=2.0)
    ap.add_argument("--ar_temperature", type=float, default=1.0,
                    help="diff head 采样温度（<1 缩小初始噪声，抑制逐帧抖动）")
    ap.add_argument("--context_frames",  type=int, default=24)
    ap.add_argument("--context_overlap", type=int, default=4)
    ap.add_argument(
        "--seed_mode", type=str, default="ref", choices=["ref", "zero"],
        help="第一帧 seed: ref=参考图 motion emb; zero=全0 neutral",
    )
    ap.add_argument("--no_gt", action="store_true", help="不渲染 GT 对照视频")
    return ap.parse_args()


# ===========================================================================
#  Worker: 每个 GPU 一个进程
# ===========================================================================
def run_worker(rank: int, world_size: int, gpu_ids: List[int],
               args: argparse.Namespace, all_samples: List[dict]):
    """
    rank:         本进程 id (0..world_size-1)
    world_size:   总进程数, 等于 len(gpu_ids)
    gpu_ids:      物理 GPU id 列表
    all_samples:  主进程已扫描好的全部样本元数据
    """
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    tag = f"[GPU{gpu_id}|rank{rank}/{world_size}]"

    # ---- 各进程独立设置 random seed (保留同样的渲染随机性) ----
    # 注: random.randint 在 pipeline 里被用作 context offset; 各 rank 独立没问题。
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # ---- 解析步数列表 ----
    steps_list = [int(s.strip()) for s in args.ar_sampling_steps.split(",") if s.strip()]
    if not steps_list:
        raise ValueError("--ar_sampling_steps 为空")

    weight_dtype = torch.float16
    ar_dtype     = torch.bfloat16
    config = OmegaConf.load(args.config)

    print(f"{tag} 加载文本/音频编码器 ...")
    tokenizer, text_encoder, audio_processor, audio_encoder = \
        load_encoders(config, device)

    print(f"{tag} 加载 X-Nemo 解码器 pipeline ...")
    pipe = load_xnemo_pipeline(config, device, weight_dtype)

    # ---- 切分样本 (步长式切片, 负载较均匀) ----
    my_samples = all_samples[rank::world_size]
    print(f"{tag} 分到 {len(my_samples)}/{len(all_samples)} 个样本: "
          f"{[s['name'] for s in my_samples]}")

    report: List[dict] = []

    for s in my_samples:
        name = s["name"]
        print(f"\n{tag} {'='*50}")
        print(f"{tag} Sample: {name}")
        print(f"{tag} {'='*50}")

        # ---- 参考图 ----
        ref_path = args.ref_image or s["ref_frame"]
        if ref_path is None:
            print(f"{tag}   [Skip] 无参考图")
            continue
        ref_image_pil = Image.open(ref_path).convert("RGB").resize((args.W, args.H))
        ref_pose_pil  = ref_image_pil

        # ---- 特征提取 (sample 内复用) ----
        print(f"{tag}   特征提取 ...")
        feats = extract_features(
            s, tokenizer, text_encoder, audio_processor, audio_encoder, device)

        # ---- ref seed ----
        seed_frame = None
        if args.seed_mode == "ref":
            seed_frame = compute_ref_motion_emb(pipe, ref_pose_pil, device, ar_dtype)

        # ---- GT motion + GT 视频 (sample 内只渲染一次) ----
        gt_motion = None
        video_gt_path = None
        if s["pose_path"] and os.path.exists(s["pose_path"]):
            gt_motion = load_motion_gt(s["pose_path"]).to(device, dtype=ar_dtype)
            if not args.no_gt:
                sample_out_dir = os.path.join(args.output_dir, name)
                os.makedirs(sample_out_dir, exist_ok=True)
                video_gt_path = os.path.join(sample_out_dir, "gt.mp4")
                print(f"{tag}   渲染 GT 视频 (一次性) ...")
                gen_gt = torch.Generator(device=device); gen_gt.manual_seed(args.seed)
                video_gt = render(pipe, ref_image_pil, ref_pose_pil,
                                  gt_motion.unsqueeze(0), args, gen_gt)
                save_videos_grid(video_gt, video_gt_path, n_rows=1, fps=FPS)
                mux_audio_to_video(video_gt_path, s["audio_path"])
                print(f"{tag}   GT video -> {video_gt_path}")

        # ----------------------------------------------------------------
        #  扫各 AR sampling steps
        # ----------------------------------------------------------------
        for n_steps in steps_list:
            print(f"\n{tag}   --- AR sampling steps = {n_steps} ---")

            print(f"{tag}   加载 AR 模型 (steps={n_steps}) ...")
            ar_model = load_ar_model(config, args.ar_ckpt, device, ar_dtype, n_steps)

            ar_motion = generate_ar_motion(
                feats, ar_model, device, ar_dtype,
                args.ar_cfg_audio, args.ar_cfg_text,
                seed_frame=seed_frame, use_lcm=config.use_lcm,
                temperature=args.ar_temperature,
            )
            print(f"{tag}   AR motion: {tuple(ar_motion.shape)}  "
                  f"std={ar_motion.float().std(0).mean().item():.4f}")

            gen_ar = torch.Generator(device=device); gen_ar.manual_seed(args.seed)
            video_ar = render(pipe, ref_image_pil, ref_pose_pil,
                              ar_motion.unsqueeze(0), args, gen_ar)

            step_dir = os.path.join(args.output_dir, name)
            os.makedirs(step_dir, exist_ok=True)
            video_ar_path = os.path.join(step_dir, f"{n_steps}steps_ar.mp4")
            save_videos_grid(video_ar, video_ar_path, n_rows=1, fps=FPS)
            mux_audio_to_video(video_ar_path, s["audio_path"])
            print(f"{tag}   AR video -> {video_ar_path}")

            entry = {
                "name":      name,
                "ar_steps":  n_steps,
                "ar_frames": int(ar_motion.shape[0]),
                "video_ar":  video_ar_path,
                "gpu_id":    gpu_id,
            }

            if gt_motion is not None:
                m = compute_metrics(ar_motion.float().cpu(), gt_motion.float().cpu())
                entry["metrics_ar_vs_gt"] = m
                if video_gt_path is not None:
                    entry["video_gt"] = video_gt_path
                print(f"{tag}   motion 对照  MSE={m['mse']:.5f}  "
                      f"cos={m['cos_sim']:.4f}  std_ratio={m['std_ratio']:.4f}  "
                      f"(T={m['frames']})")

            report.append(entry)

            del ar_model
            torch.cuda.empty_cache()

    # ---- 写入分片 report ----
    partial_path = os.path.join(args.output_dir, f".partial_report_rank{rank}.json")
    with open(partial_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{tag} 完成, 分片报告 -> {partial_path}")


# ===========================================================================
#  Main: 启动子进程 + 合并结果
# ===========================================================================
def main():
    args = parse_args()

    # ---- 解析 GPU 列表 ----
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus 为空, 请提供至少一张 GPU, 如 '0' 或 '0,1,2,3'")
    world_size = len(gpu_ids)

    # ---- 解析步数, 仅用于最后 summary 打印; 真正使用在 worker 内 ----
    steps_list = [int(s.strip()) for s in args.ar_sampling_steps.split(",") if s.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 主进程一次性收集样本, 避免每个子进程重复扫盘且顺序不一致 ----
    if args.sample_ids:
        want = set(x.strip() for x in args.sample_ids.split(",") if x.strip())
        samples = [s for s in collect_samples(args.test_dir, None) if s["name"] in want]
    else:
        samples = collect_samples(args.test_dir, args.num_samples)
    print(f"[Main] 总样本数 {len(samples)}, 使用 {world_size} 张 GPU: {gpu_ids}")
    print(f"[Main] AR sampling steps sweep: {steps_list}")

    if world_size == 1:
        # 单卡直接跑, 不走 spawn (方便调试和挂断)
        run_worker(0, 1, gpu_ids, args, samples)
    else:
        # 多卡: spawn 子进程
        # CUDA 初始化需要 'spawn' 启动方式
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        mp.spawn(
            run_worker,
            args=(world_size, gpu_ids, args, samples),
            nprocs=world_size,
            join=True,
        )

    # ----------------------------------------------------------------
    #  合并 partial reports
    # ----------------------------------------------------------------
    report: List[dict] = []
    for rank in range(world_size):
        partial_path = os.path.join(args.output_dir, f".partial_report_rank{rank}.json")
        if os.path.exists(partial_path):
            with open(partial_path) as f:
                report.extend(json.load(f))
            os.remove(partial_path)
        else:
            print(f"[Main][warn] 缺失分片: {partial_path}")

    # 按样本名 + 步数排序, 输出更可读
    def _sort_key(e):
        # 数字目录名也能排
        try:
            n = int(e["name"])
        except Exception:
            n = e["name"]
        return (n, e.get("ar_steps", 0))
    report.sort(key=_sort_key)

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------
    #  Summary
    # ----------------------------------------------------------------
    has_metrics = [e for e in report if "metrics_ar_vs_gt" in e]
    if has_metrics:
        print(f"\n{'='*60}")
        print("[Summary] AR vs GT motion metrics (按步数平均)")
        print(f"{'='*60}")
        print(f"  {'steps':>8}  {'MSE':>10}  {'MAE':>10}  {'cos_sim':>10}  {'std_ratio':>10}  {'#samples':>9}")
        print(f"  {'-'*70}")
        for n_steps in steps_list:
            rows = [e["metrics_ar_vs_gt"] for e in has_metrics if e["ar_steps"] == n_steps]
            if not rows:
                continue
            print(
                f"  {n_steps:>8}"
                f"  {np.mean([r['mse'] for r in rows]):>10.5f}"
                f"  {np.mean([r['mae'] for r in rows]):>10.5f}"
                f"  {np.mean([r['cos_sim'] for r in rows]):>10.4f}"
                f"  {np.mean([r['std_ratio'] for r in rows]):>10.4f}"
                f"  {len(rows):>9d}"
            )

    print(f"\n[Done] 输出目录: {args.output_dir}")
    print(f"[Done] 报告: {report_path}")
    print("\n输出结构:")
    print("  output_dir/")
    print("    {sample_name}/")
    print("      gt.mp4              <- GT 天花板 (各步数共用)")
    for n in steps_list:
        print(f"      {n}steps_ar.mp4      <- AR steps={n} 生成视频")
    print("    report.json")


if __name__ == "__main__":
    main()
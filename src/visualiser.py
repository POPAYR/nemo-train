"""
xnemo_visualiser.py
===================
训练中途可视化（所有 rank 分片）：用「当前训练中的 AR 权重」走 X-Nemo 推理链路
（AR LCM 一步 rollout → 注入 X-Nemo denoising/reference UNet → VAE decode），
保存 mp4 并并入音频。

本版改动（对齐训练/部署）：
  [改] AR motion 生成从 generate(num_sampling_steps=20, use_ddim=True) 的多步采样
       改为与训练完全一致的 LCM 一步 rollout（rollout_lcm_infer）。
       原因：训练态优化的是"一步生成"的 motion 质量；若可视化用 20 步多步采样，
       看到的 motion 与训练/部署(一步)不一致，会误判训练效果。
  [改] gen_norm 默认值 True -> False，与训练脚本(默认 False)统一；并以 config 的
       generate_returns_normalized 为准（config 已显式写死，避免两边相反）。
"""

import json
import math
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import random

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import moviepy.editor as mpy

from diffusers import AutoencoderKLTemporalDecoder, DDIMScheduler
from src.pipelines.context import get_context_scheduler
from src.pipelines.pipeline_pose2vid_motenc_long import Pose2VideoPipelineOutput
import deepspeed.comm as dist

from transformers import Wav2Vec2Processor
from utils.processor import get_windowed_audio

from model.armodel import SelfAttnKVCache, CrossAttnKVCache

AUDIO_SR = 16000
BBOX_PARAM_DIM = 3


# ---------------- audio helpers ----------------
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


# ---------------- LCM one-step diff head（与训练脚本一致）----------------
def lcm_1step(diff_net, z, alphas_cp, T_max, noise=None):
    B, D = z.shape
    if noise is None:
        noise = torch.randn(B, D, device=z.device, dtype=z.dtype)
    t = torch.full((B,), T_max - 1, device=z.device)
    v = diff_net(noise, t.float(), c=z)
    a = alphas_cp[t].view(-1, 1).to(z.dtype)
    return a.sqrt() * noise - (1 - a).sqrt() * v


@torch.no_grad()
def rollout_lcm_infer(student, seed_motion_norm, text_emb, audio_emb,
                      local_audio_full, alphas_cp, T_max, T, autocast_factory):
    """与训练 rollout_lcm 同构的推理态 rollout：全程 self-forcing，LCM 一步/帧。
    seed_motion_norm: [1,1,D] 归一化空间的第 0 帧种子。
    返回 motion [1, T, D]（归一化空间；调用方按 gen_norm 决定是否 denorm）。"""
    with autocast_factory():
        fusion_latents = student.fusion_net(text_emb, audio_emb)
        local_proj = student.audio_proj(local_audio_full)
        self_caches = [SelfAttnKVCache() for _ in student.layers]
        fusion_caches = [CrossAttnKVCache() for _ in student.layers]

        cur = seed_motion_norm                          # [1,1,D]
        preds = [cur[:, 0]]                             # 含种子帧，凑齐 T 帧
        for t in range(T - 1):
            local_t = local_proj[:, t:t + 1]
            h = student.motion_proj(cur)
            for layer, sc, fc in zip(student.layers, self_caches, fusion_caches):
                h = layer.forward_cached(h, fusion_latents, local_t, sc, fc, student.max_len)
            z = student.norm(h[:, -1])
            pred = lcm_1step(student.diffloss.net, z, alphas_cp, T_max)   # [1, D]
            preds.append(pred)
            cur = pred.unsqueeze(1)
    return torch.stack(preds, dim=1)                    # [1, T, D]


# ===========================================================================
#  视频保存（并入音频）
# ===========================================================================
def save_video_with_audio(frames: List[Image.Image], path: str,
                          audio_path: Optional[str] = None, fps: int = 25):
    clips = [np.array(f) for f in frames]
    video = mpy.ImageSequenceClip(clips, fps=fps)
    if audio_path and os.path.exists(audio_path):
        audio = mpy.AudioFileClip(audio_path)
        dur = min(video.duration, audio.duration)
        video = video.subclip(0, dur).set_audio(audio.subclip(0, dur))
        video.write_videofile(path, fps=fps, codec="libx264", audio_codec="aac", logger=None)
        audio.close()
    else:
        video.write_videofile(path, fps=fps, codec="libx264", logger=None)
    video.close()


# ===========================================================================
#  可视化器
# ===========================================================================
class XNemoARVisualiser:
    def __init__(self, config, trainer, device, dtype):
        self.cfg = config
        self.vis_cfg = config.visualisation
        self.device = device
        self.dtype = dtype
        self.t = trainer
        self.spf = config.data.sr // config.data.fps
        self.window_frames = config.data.get("window_frames", 5)
        self.vae_scale = config.get("vae_scale", 0.18215)
        self.motion_token_shape = tuple(config.get("motion_token_shape", [32, 16]))

        stats = torch.load(config.data.data_stats_path, map_location="cpu")
        self.mean = stats["mean"].float().reshape(-1).to(device)
        self.std = stats["std"].float().reshape(-1).to(device)
        # [修] 默认 False，与训练脚本统一；以 config 显式值为准
        self.gen_norm = config.get("generate_returns_normalized", False)

        # LCM 一步采样常量（与训练同源）
        self.alphas_cp = self.t.motion_predictor.diffloss.train_scheduler.alphas_cumprod.to(
            device).float()
        self.T_max = self.t.motion_predictor.diffloss.train_scheduler.config.num_train_timesteps

        self._load_frozen()
        self._load_val_samples()

    def _load_frozen(self):
        cfg = self.cfg
        from omegaconf import OmegaConf
        infer_config = OmegaConf.load(cfg.inference_config)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            cfg.vae_path).to(self.device, dtype=torch.float32).eval()
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)
        from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
        self.motion_encoder = MotEncoder().to(self.device, dtype=self.dtype).eval()
        if cfg.get("motion_encoder_path", None):
            self.motion_encoder.load_state_dict(
                torch.load(cfg.motion_encoder_path, map_location="cpu"), strict=True)
        print("[Vis] frozen comps loaded (VAE, DDIM, motion_encoder)")

    def _load_val_samples(self):
        self.samples = []
        for e in self.vis_cfg.val_samples:
            try:
                self.samples.append(self._parse(e))
            except Exception as exc:
                print(f"[Vis] skip {e.get('name', '?')}: {exc}")
        print(f"[Vis] loaded {len(self.samples)} val samples")

    def _parse(self, e):
        cap = e.get("caption", "A person speaking")
        if isinstance(cap, str) and cap.endswith(".json"):
            with open(cap) as f:
                data = json.load(f)
            cap = list(data.values())[0] if isinstance(data, dict) else data[0]
        gt = None
        if e.get("motion_emb_path", None) and os.path.exists(e.motion_emb_path):
            gt = torch.load(e.motion_emb_path, map_location="cpu")
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            gt = gt.reshape(gt.shape[0], -1).float()
        return {
            "name": e.get("name", Path(e.ref_image_path).stem),
            "ref_image": Image.open(e.ref_image_path).convert("RGB"),
            "audio_path": e.audio_path,
            "caption": cap,
            "gt_motion": gt,
        }

    # --------------------- 特征 + AR 生成（LCM 一步，对齐训练）---------------------
    @torch.no_grad()
    def _gen_ar_motion(self, sample):
        cfg, vis = self.cfg, self.vis_cfg
        W = H = vis.get("height", 512)

        from transformers import AutoTokenizer
        if not hasattr(self, "_tok"):
            self._tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path)
        tok = self._tok(sample["caption"], padding="max_length", truncation=True,
                        max_length=cfg.data.get("text_max_len", 128),
                        return_tensors="pt")
        text_token = tok.input_ids.to(self.device)
        text_mask  = tok.attention_mask.to(self.device)
        text_emb = self.t.text_encoder(input_ids=text_token,
                                    attention_mask=text_mask).last_hidden_state
        text_emb = text_emb * text_mask.unsqueeze(-1)

        audio, _ = librosa.load(sample["audio_path"], sr=AUDIO_SR)
        T = min(len(audio) // self.spf, vis.get("max_frames", 10_000))
        audio = audio[: T * self.spf]
        audio_clip = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        wav = normalize_wav(audio_clip)
        audio_emb = self.t.audio_encoder(wav).last_hidden_state
        frame_feat = align_to_frames(audio_emb, T)
        local_audio_emb = build_local_window(frame_feat, half=self.window_frames // 2)

        # seed：raw 空间，需 normalize 到 rollout 的归一化空间
        seed_raw = self._ref_motion_seed(sample["ref_image"], W, H)   # [1,1,512] raw
        seed_norm = (seed_raw - self.mean.view(1, 1, -1)) / (self.std.view(1, 1, -1) + 1e-6)

        # LCM 一步 rollout（与训练 rollout_lcm 同构）
        motion_norm = rollout_lcm_infer(
            self.t.motion_predictor, seed_norm, text_emb, audio_emb, local_audio_emb,
            self.alphas_cp, self.T_max, T,
            autocast_factory=lambda: torch.cuda.amp.autocast(dtype=self.dtype),
        )                                                  # [1,T,512] 归一化空间

        # 喂 X-Nemo decoder 需要 raw 空间：按 gen_norm 决定
        if self.gen_norm:
            motion = motion_norm
        else:
            motion = motion_norm * (self.std.view(1, 1, -1) + 1e-6) + self.mean.view(1, 1, -1)
        return motion, T

    @torch.no_grad()
    def _ref_motion_seed(self, ref_pil, W, H):
        from diffusers.image_processor import VaeImageProcessor
        if not hasattr(self, "_cond_proc"):
            self._cond_proc = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=True)
        bbox = torch.ones((1, BBOX_PARAM_DIM), device=self.device, dtype=self.dtype)
        bbox[:, :2] *= 0
        cond = self._cond_proc.preprocess(ref_pil, height=224, width=224).to(self.device, dtype=self.dtype)
        emb = self.motion_encoder(cond, bbox)
        return emb.reshape(1, 1, -1).to(self.dtype)        # [1,1,512] raw

    # --------------------- 渲染（注入 X-Nemo decoder）---------------------
    @torch.no_grad()
    def _render(self, motion_raw, ref_pil, T):
        from src.models.mutual_self_attention import ReferenceAttentionControl
        vis = self.vis_cfg
        W = H = vis.get("height", 512)
        dev, dt = self.device, self.dtype
        t = self.t
        steps = vis.get("num_inference_steps", 25)
        guidance = vis.get("guidance_scale", 2.5)
        do_cfg = guidance > 1.0

        self.scheduler.set_timesteps(steps, device=dev)
        timesteps = self.scheduler.timesteps

        try:
            t.ref_writer.clear(); t.ref_reader.clear()
        except Exception:
            pass

        writer = ReferenceAttentionControl(
            t.reference_unet, do_classifier_free_guidance=do_cfg,
            mode="write", batch_size=1, fusion_blocks="full")
        reader = ReferenceAttentionControl(
            t.denoising_unet, do_classifier_free_guidance=do_cfg,
            mode="read", batch_size=1, fusion_blocks="full")

        try:
            from diffusers.image_processor import VaeImageProcessor
            from transformers import CLIPImageProcessor
            if not hasattr(self, "_clip_proc"):
                self._clip_proc = CLIPImageProcessor()
            clip_in = self._clip_proc.preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values
            clip_emb = t.image_encoder(clip_in.to(dev, dtype=t.image_encoder.dtype)).image_embeds.unsqueeze(1).to(dt)
            if do_cfg:
                uncond_clip = torch.zeros_like(clip_emb)
                clip_emb_in = torch.cat([uncond_clip, clip_emb], dim=0)
            else:
                clip_emb_in = clip_emb

            if not hasattr(self, "_ref_proc"):
                self._ref_proc = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True)
            ref_tensor = self._ref_proc.preprocess(ref_pil, height=H, width=W).to(self.vae.device, dtype=self.vae.dtype)
            ref_latents = self.vae.encode(ref_tensor).latent_dist.mean * self.vae_scale

            lat_h, lat_w = H // 8, W // 8
            repeated = ref_latents.unsqueeze(2).repeat(1, 1, T, 1, 1)
            import torchvision.transforms as TT
            blur = TT.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
            rep_flat = repeated.permute(0, 2, 1, 3, 4).reshape(T, 4, lat_h, lat_w)
            rep_blur = blur(rep_flat).reshape(1, T, 4, lat_h, lat_w).permute(0, 2, 1, 3, 4)
            noise = torch.randn_like(rep_blur)
            noisy_first = []
            for f in range(T):
                noisy_first.append(self.scheduler.add_noise(
                    rep_blur[:, :, f:f+1], noise[:, :, f:f+1], timesteps[:1]))
            latents = torch.cat(noisy_first, dim=2).to(dtype=dt)

            motion_tokens_full = motion_raw.to(dev, dtype=dt).reshape(1, T, *self.motion_token_shape)

            neg_motion = None
            if do_cfg:
                ref_bbox = torch.ones((1, BBOX_PARAM_DIM), device=dev, dtype=dt)
                ref_bbox[:, :2] *= 0
                ref_cond = self._cond_proc.preprocess(ref_pil, height=224, width=224).to(dev, dtype=dt)
                neg_motion = self.motion_encoder(ref_cond, ref_bbox).to(dt)

            context_scheduler = get_context_scheduler("uniform")
            ctx_frames = vis.get("context_frames", 24)
            ctx_overlap = vis.get("context_overlap", 4)

            ref_lat_in = ref_latents.repeat(2 if do_cfg else 1, 1, 1, 1)
            t.reference_unet(ref_lat_in, torch.zeros_like(timesteps[:1]),
                            encoder_hidden_states=clip_emb_in, return_dict=False)
            reader.update(writer)

            for ti, tt in (enumerate(timesteps)):
                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_cfg else 1), *latents.shape[1:]),
                    device=dev, dtype=dt)
                counter = torch.zeros((1, 1, T, 1, 1), device=dev, dtype=dt)
                offset = random.randint(0, ctx_frames - 1)
                context_queue = list(context_scheduler(0, steps, T, ctx_frames, 1, ctx_overlap, True, offset))

                for ctx in context_queue:
                    lat_in = latents[:, :, ctx]
                    lat_in = lat_in.repeat(2 if do_cfg else 1, 1, 1, 1, 1)
                    lat_in = self.scheduler.scale_model_input(lat_in, tt)
                    mtok = motion_tokens_full[:, ctx]

                    if do_cfg:
                        neg_mtok = neg_motion.unsqueeze(1).expand_as(mtok)
                        mtok_in = torch.cat([neg_mtok, mtok], dim=0)
                    else:
                        mtok_in = mtok

                    pred = t.denoising_unet(
                        lat_in, tt,
                        encoder_hidden_states=[clip_emb_in, mtok_in],
                        pose_cond_fea=None, return_dict=False)[0]

                    noise_pred[:, :, ctx] += pred
                    counter[:, :, ctx] += 1

                noise_pred = noise_pred / counter
                if do_cfg:
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + guidance * (cond - uncond)
                latents = self.scheduler.step(noise_pred, tt, latents).prev_sample

        finally:
            try: reader.clear()
            except: pass
            try: writer.clear()
            except: pass
            try: t.ref_reader.clear(); t.ref_writer.clear()
            except: pass

        lat = latents.permute(0, 2, 1, 3, 4).reshape(T, 4, lat_h, lat_w) / self.vae_scale
        lat = lat.to(self.vae.dtype)
        decode_chunk = self.vis_cfg.get("decode_chunk_size", 8)
        imgs = []
        for i in range(0, T, decode_chunk):
            chunk = lat[i:i+decode_chunk]
            n = chunk.shape[0]
            out = self.vae.decode(chunk, num_frames=n).sample
            out = ((out / 2 + 0.5).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
            for k in range(n):
                imgs.append(Image.fromarray(out[k].transpose(1, 2, 0)))
        return imgs

    # --------------------- 入口 ---------------------
    @torch.no_grad()
    def run(self, step, vis_dir, samples=None):
        if samples is None:
            samples = self.samples
        if not samples:
            return
        step_dir = Path(vis_dir) / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        fps = self.vis_cfg.get("fps", 25)

        self.t.motion_predictor.eval()
        try:
            for s in samples:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    motion, T = self._gen_ar_motion(s)
                    frames_ar = self._render(motion, s["ref_image"], T)
                save_video_with_audio(frames_ar, str(step_dir / f"{s['name']}_AR.mp4"),
                                    s["audio_path"], fps)
        finally:
            self.t.motion_predictor.train()
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Vis] rank {rank} step {step}: {len(samples)} done -> {step_dir}")
# *************************************************************************
# Phase 1: 因果流式推理管线 (DECODER_DISTILL_PLAN.md §Phase 1 余项3)
# 把因果 UNet + reference bank(缓存一次) + 逐帧 motion + TAESD 串成真正出视频的流式管线。
# 逐 block 自回归：每个 block 跑 N 步 DDIM 去噪（KV-cache 读干净历史 + 当前噪声），
# block 去噪干净后用一次 commit 前向把干净 K/V 写入缓存；再解码该 block。
# *************************************************************************
import time
import numpy as np
import torch
from einops import rearrange

from .pipeline_pose2vid_motenc_long import Pose2VideoPipeline
from ..models.mutual_self_attention import ReferenceAttentionControl
from ..models.temporal_causal import TemporalCausalControl

BBOX_PARAM_DIM = 3
MOTION_TOKEN_SHAPE = (32, 16)


class CausalStreamingPipeline(Pose2VideoPipeline):

    @torch.no_grad()
    def stream(
        self,
        ref_image,                 # PIL
        motion_latents,            # [1, T, 512]  (已归一化空间, 与 decoder 训练一致)
        ref_pose_image=None,       # PIL，CFG 的 neg_motion 用；None 则不可 CFG
        block_size=8, window=24,
        num_inference_steps=25, guidance_scale=2.5,
        width=512, height=512,
        decoder="taesd", taesd=None,    # "taesd" 需传入 AutoencoderTiny；否则用 SVD
        generator=None, init_noise=None, verbose=True,
    ):
        # init_noise: 可选 [1,4,T,h,w]，逐 block 取片，使 causal vs bidir 用完全相同初始噪声做公平对比。
        device = self._execution_device
        do_cfg = guidance_scale > 1.0 and ref_pose_image is not None
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        ud = self.denoising_unet.dtype
        T = motion_latents.shape[1]
        latent_h, latent_w = height // 8, width // 8

        # ---- CLIP 图像嵌入 ----
        clip_pix = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt").pixel_values
        clip_emb = self.image_encoder(
            clip_pix.to(device, self.image_encoder.dtype)).image_embeds.unsqueeze(1)
        clip_in = torch.cat([torch.zeros_like(clip_emb), clip_emb]) if do_cfg else clip_emb

        # ---- reference attention（整片只写一次） ----
        writer = ReferenceAttentionControl(self.reference_unet, do_classifier_free_guidance=do_cfg,
                                           mode="write", batch_size=1, fusion_blocks="full")
        reader = ReferenceAttentionControl(self.denoising_unet, do_classifier_free_guidance=do_cfg,
                                           mode="read", batch_size=1, fusion_blocks="full")
        ref_t = self.ref_image_processor.preprocess(ref_image, height=height, width=width).to(
            dtype=self.vae.dtype, device=device)
        ref_latents = self.vae.encode(ref_t).latent_dist.mean * 0.18215
        t0 = torch.zeros_like(timesteps[0])
        writer.clear()
        self.reference_unet(ref_latents.repeat(2 if do_cfg else 1, 1, 1, 1), t0,
                            encoder_hidden_states=clip_in, return_dict=False)
        reader.update(writer, dtype=self.denoising_unet.dtype)

        # ---- CFG 的 neg motion ----
        neg_motion = None
        if do_cfg:
            bbox = torch.ones((1, BBOX_PARAM_DIM), device=device, dtype=self.motion_encoder.dtype)
            bbox[:, :2] *= 0
            ref_cond = self.cond_image_processor.preprocess(
                ref_pose_image, height=224, width=224).to(device=device, dtype=self.motion_encoder.dtype)
            neg_motion = self.motion_encoder(ref_cond, bbox)            # [1,32,16]

        # ---- 因果控制 ----
        ctrl = TemporalCausalControl(self.denoising_unet, block_size=block_size, window=window)
        ctrl.set_mode("stream"); ctrl.reset_cache()

        motion_latents = motion_latents.to(device, ud)
        out_frames = []
        rng = range(0, T, block_size)
        _bench = getattr(self, "_bench_block_times", None)
        for s in rng:
            if _bench is not None:
                torch.cuda.synchronize(device); _bench_t0 = time.perf_counter()
            k = min(block_size, T - s)
            mot = motion_latents[:, s:s + k].reshape(1, k, *MOTION_TOKEN_SHAPE)
            if do_cfg:
                neg = neg_motion.unsqueeze(1).expand(1, k, *MOTION_TOKEN_SHAPE)
                mot_in = torch.cat([neg, mot], dim=0)
            else:
                mot_in = mot

            if init_noise is not None:
                z = init_noise[:, :, s:s + k].to(device, ud)
            else:
                z = torch.randn((1, 4, k, latent_h, latent_w), generator=generator, device=device, dtype=ud)

            # ---- 逐 block N 步去噪（commit=False，只读干净历史缓存） ----
            ctrl.set_offset(s); ctrl.set_commit(False)
            for t in timesteps:
                zin = torch.cat([z] * 2) if do_cfg else z
                zin = self.scheduler.scale_model_input(zin, t)
                eps = self.denoising_unet(zin, t, encoder_hidden_states=[clip_in, mot_in],
                                          pose_cond_fea=None, return_dict=False)[0]
                if do_cfg:
                    e_u, e_c = eps.chunk(2)
                    eps = e_u + guidance_scale * (e_c - e_u)
                z = self.scheduler.step(eps, t, z).prev_sample

            # ---- commit：用干净 z 在 t=0 跑一次，把干净 K/V 写入缓存供后续 block ----
            ctrl.set_commit(True)
            zin = torch.cat([z] * 2) if do_cfg else z
            self.denoising_unet(zin, t0, encoder_hidden_states=[clip_in, mot_in],
                                pose_cond_fea=None, return_dict=False)

            out_frames.append(self._decode_block(z, decoder, taesd))
            if _bench is not None:
                torch.cuda.synchronize(device)
                _bench.append(time.perf_counter() - _bench_t0)
            if verbose:
                print(f"  [stream] block {s}-{s+k}/{T} done", flush=True)

        reader.clear(); writer.clear(); ctrl.disable()
        video = torch.cat(out_frames, dim=2)        # [1,3,T,H,W] in [0,1]
        return video

    def _decode_block(self, z, decoder, taesd):
        # z: [1,4,k,64,64] scaled latent (UNet 空间)
        if decoder == "taesd" and taesd is not None:
            zf = rearrange(z, "b c f h w -> (b f) c h w").to(taesd.dtype)
            img = taesd.decode(zf).sample              # AutoencoderTiny 输出在 [-1,1]
            img = (img / 2 + 0.5).clamp(0, 1).float().unsqueeze(0).permute(0, 2, 1, 3, 4)  # → [0,1] [1,3,k,H,W]
            return img.cpu()
        else:
            # SVD 时序 decoder：一次解多帧显存暴涨(8帧≈58GB)，按 chunk=2 解以限显存
            vid = self.decode_latents_svd(z, decode_chunk_size=2)   # numpy [1,3,k,H,W] in [0,1]
            return torch.from_numpy(vid)

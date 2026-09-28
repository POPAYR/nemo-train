"""
Phase 1 验收(可视): 因果流式 vs 双向 解码同一段真实 motion, 出视频对比。
- 同一 ref + 同一 motion + 同一初始噪声(init_noise) -> 差异 = 纯因果性。
- causal:  block_size=8 (逐 block KV-cache 流式)
- bidir:   block_size=T (单块=窗口内全双向, 近似原模型)
用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/test_causal_stream_video.py
"""
import sys, os, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder, AutoencoderTiny
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
from src.pipelines.causal_streaming_pipeline import CausalStreamingPipeline
from src.utils.util import save_videos_grid

dev = torch.device("cuda:0"); dt = torch.float16
cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")
ic = OmegaConf.load(cfg.inference_config)

# ---- 构建模型 ----
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt).eval()
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
motenc = MotEncoder().to(dev, dt).eval()
motenc.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "motion_encoder"), map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
sched = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()

pipe = CausalStreamingPipeline(vae=vae, image_encoder=imgenc, reference_unet=refu,
                               denoising_unet=denu, motion_encoder=motenc, scheduler=sched).to(dev, dt)

# ---- 真实样本 ----
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
name = "M003_video_down_angry_level_1_001"
T = 32
ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
mot = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float()  # [1,Tfull,32,16]
mot = mot[:, :T].reshape(1, T, -1).to(dev, dt)                                  # [1,T,512]
print(f"[sample] {name}  ref=512x512  motion={tuple(mot.shape)}")

STEPS, CFGV, BLK, WIN = 20, 1.0, 8, 0
g = torch.Generator(device=dev); g.manual_seed(1234)
init_noise = torch.randn((1, 4, T, 64, 64), generator=g, device=dev, dtype=dt)

outdir = "/media/ps/ssd5/ayr/x-nemo-inference/output/causal_stream"; os.makedirs(outdir, exist_ok=True)

print("[run] causal (block=8) ...")
vid_causal = pipe.stream(ref_pil, mot, block_size=BLK, window=WIN, num_inference_steps=STEPS,
                         guidance_scale=CFGV, decoder="taesd", taesd=taesd, init_noise=init_noise)
print("[run] bidir (block=T) ...")
vid_bidir = pipe.stream(ref_pil, mot, block_size=T, window=WIN, num_inference_steps=STEPS,
                        guidance_scale=CFGV, decoder="taesd", taesd=taesd, init_noise=init_noise)

save_videos_grid(vid_causal, f"{outdir}/{name}_causal_blk8.mp4", n_rows=1, fps=25)
save_videos_grid(vid_bidir, f"{outdir}/{name}_bidir.mp4", n_rows=1, fps=25)
# 并排
both = torch.cat([vid_bidir, vid_causal], dim=4)  # 横向拼 [bidir | causal]
save_videos_grid(both, f"{outdir}/{name}_bidir_vs_causal.mp4", n_rows=1, fps=25)

d = (vid_causal - vid_bidir).abs()
print(f"\n[像素差 causal vs bidir]  mean={d.mean().item():.4f}  max={d.max().item():.4f}  (像素∈[0,1])")
print(f"[done] 视频 -> {outdir}/  ({name}_causal_blk8.mp4 / _bidir.mp4 / _bidir_vs_causal.mp4)")

"""对比 monolithic self_forcing_rollout vs stream_pipeline 逐block，定位长rollout噪声来源。
同一真输入(GT pose_embed motion, 160帧)。GPU 由 CUDA_VISIBLE_DEVICES 指定。"""
import sys, os, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.rollout import self_forcing_rollout

dev = torch.device("cuda:0")
CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
n = "M023_video_right_30_fear_level_3_010"
F_ = 160
OUT = "/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"

M = DMD2Models(dev, gen_ckpt=None, block_size=8)
rk = torch.load(CKPT, map_location="cpu")
sd = rk.get("generator_ema") or rk["generator"]
M.generator.load_state_dict(sd, strict=False)
print(f"[loaded ema @ step {rk['step']}]")
cfg = OmegaConf.load(DEC_CFG)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, M.dt).eval()
clip_proc = CLIPImageProcessor()

ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = M.image_encoder(clip_proc.preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, M.dt)).image_embeds.unsqueeze(1)
ref_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[0:1].to(dev, M.dt)
motion = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float()
motion = motion.reshape(motion.shape[0] if motion.dim()==2 else motion.shape[1], -1)[:F_].reshape(1, F_, 32, 16).to(dev, M.dt)
g = torch.Generator(device=dev).manual_seed(1234)
noise = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=M.dt)
dsl = [999, 749, 499, 249]

@torch.no_grad()
def decode_and_save(x0, tag):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)  # [F,3,H,W]
    for f in [0, 40, 80, 120]:
        if f < v.shape[0]:
            arr = (v[f].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            Image.fromarray(arr).save(f"{OUT}/{tag}_f{f}.png")
    print(f"[{tag}] x0.std={x0.float().std():.3f} saved f0/40/80/120")

# A) monolithic
M.set_reference(ref_lat, clip, 1)
M.gen_causal.set_window(0)
x0_mono, _ = self_forcing_rollout(M, noise, clip, motion, dsl, block_size=8, grad_window=None, full_steps=True)
decode_and_save(x0_mono, "mono")

# B) 我的 VideoStream 逐 block
from src.distill.stream_pipeline import VideoStream
vs = VideoStream(M, vae, dev, M.dt, denoising_step_list=dsl, block_size=8, causal_window=0)
# 直接喂 GT motion(反推成 [F,512])，绕过 AR
mot_flat = motion.reshape(F_, 512)
# set_reference 需要 ref_latent(未乘scale) 和 ref_img；这里直接手动设,复用同一 M 的 bank
vs.clip_emb = clip
vs._lat_shape = tuple(ref_lat.shape[1:])
M.set_reference(ref_lat, clip, 1)
M.gen_causal.set_mode("stream"); M.gen_causal.set_window(0); M.gen_causal.reset_cache()
vs._offset = 0
outs = []
for b in range(F_ // 8):
    blk = mot_flat[b*8:(b+1)*8]
    outs.append(vs.render_block(blk))
x0_stream = torch.cat(outs, dim=2)
decode_and_save(x0_stream, "stream")
print("DONE")

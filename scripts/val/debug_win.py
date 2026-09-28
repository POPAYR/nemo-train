"""隔离测试:causal 解码器(stream)用 GT pose_embed motion 渲 64 帧,对比 window=0 vs 24。
排除 AR motion 变量。CUDA_VISIBLE_DEVICES 指定 GPU。"""
import sys, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.stream_pipeline import VideoStream

dev = torch.device("cuda:0")
CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
n = "M023_video_right_30_fear_level_3_010"; F_ = 64
OUT = "/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"

M = DMD2Models(dev, gen_ckpt=None, block_size=8)
rk = torch.load(CKPT, map_location="cpu")
M.generator.load_state_dict(rk["generator_ema"], strict=False)
print(f"[loaded ema step {rk['step']}]", flush=True)
cfg = OmegaConf.load(DEC_CFG)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, M.dt).eval()
clip_proc = CLIPImageProcessor()

ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = M.image_encoder(clip_proc.preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, M.dt)).image_embeds.unsqueeze(1)
ref_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[0:1].to(dev, M.dt)  # 已scaled
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float()
mo = mo.reshape(-1, 32 * 16)                    # [1,T,32,16] or [T,512] -> [T,512]
mo = mo[:F_].to(dev, M.dt)                      # [64,512]

@torch.no_grad()
def decode_save(x0, tag):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    for f in [0, 24, 48, 63]:
        if f < z.shape[0]:
            img = (vae.decode(z[f:f+1], 1).sample / 2 + 0.5).clamp(0, 1)[0]
            arr = (img.permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
            Image.fromarray(arr).save(f"{OUT}/win_{tag}_f{f}.png")
    print(f"[{tag}] x0.std={x0.float().std():.3f}", flush=True)

@torch.no_grad()
def run(window, tag):
    vs = VideoStream(M, vae, dev, M.dt, block_size=8, causal_window=window, reset_period=0)
    vs.clip_emb = clip; vs._lat_shape = tuple(ref_lat.shape[1:])
    M.set_reference(ref_lat, clip, 1)
    M.gen_causal.set_mode("stream"); M.gen_causal.set_window(window); M.gen_causal.reset_cache()
    vs._offset = 0
    outs = [vs.render_block(mo[b*8:(b+1)*8]) for b in range(F_//8)]
    decode_save(torch.cat(outs, dim=2), tag)

run(0, "w0")
run(24, "w24")
print("DONE", flush=True)

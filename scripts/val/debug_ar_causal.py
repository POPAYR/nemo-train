"""决定性隔离: 验证过的 generate_ar_motion + self_forcing_rollout(训练脚本用法) 渲 causal。
对比 GT motion。看 AR motion 喂 causal 解码器是否发散。CUDA_VISIBLE_DEVICES 指定 GPU。"""
import os, sys, torch
os.environ["AR_REPO_ROOT"] = "/media/ps/ssd5/ayr/motar"
os.environ["XNEMO_REPO_ROOT"] = "/media/ps/ssd5/ayr/x-nemo-inference"
for p in ("/media/ps/ssd5/ayr/motar", "/media/ps/ssd5/ayr/x-nemo-inference",
          "/media/ps/ssd5/ayr/x-nemo-inference/scripts/val"):
    sys.path.append(p)
import torch, librosa
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.rollout import self_forcing_rollout
from test_ar_model import (load_encoders, load_ar_model, generate_ar_motion, load_motion_gt,
                           normalize_wav, align_to_frames, build_local_window, AUDIO_SR, FPS, WINDOW_SIZE)

dev = torch.device("cuda:0"); dt = torch.bfloat16
CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
XCFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml"
AR_CKPT = ("/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD_video_v3/"
           "20260702_080807_sf_gan_v32/checkpoints/sf_gan_step_3000.pt")
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"; n = "M023_video_right_30_fear_level_3_010"; F_ = 64
OUT = "/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"
cfg = OmegaConf.load(XCFG)

M = DMD2Models(dev, gen_ckpt=None, block_size=8)
rk = torch.load(CKPT, map_location="cpu"); M.generator.load_state_dict(rk["generator_ema"], strict=False)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
ar = load_ar_model(cfg, AR_CKPT, dev, dt, 1)
tok, txte, aproc, aenc = load_encoders(cfg, dev)
print("[loaded all]", flush=True)

ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = M.image_encoder(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)), return_tensors="pt").pixel_values.to(dev,dt)).image_embeds.unsqueeze(1)
ref_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[0:1].to(dev, dt)

# ---- AR motion via 验证过的 generate_ar_motion (seed=GT首帧) ----
import glob
wavp = f"{ROOT}/audio/{n}.wav"
audio, _ = librosa.load(wavp, sr=AUDIO_SR); T=len(audio)//(AUDIO_SR//FPS); audio=audio[:T*(AUDIO_SR//FPS)]
ac = torch.from_numpy(audio).float().unsqueeze(0).to(dev)
with torch.no_grad():
    wav=normalize_wav(ac); aemb=aenc(wav).last_hidden_state
    ff=align_to_frames(aemb,T); law=build_local_window(ff,half=WINDOW_SIZE//2)
    cap=open(f"{ROOT}/emo_pose_caption/{n}.txt").read().strip() if os.path.exists(f"{ROOT}/emo_pose_caption/{n}.txt") else "a person is talking"
    tk=tok(cap,padding="max_length",truncation=True,max_length=128,return_tensors="pt")
    ti,tm=tk.input_ids.to(dev),tk.attention_mask.to(dev)
    with torch.cuda.amp.autocast(dtype=torch.float32):
        te=txte(input_ids=ti,attention_mask=tm).last_hidden_state
    te=te*tm.unsqueeze(-1)
seed=load_motion_gt(f"{ROOT}/pose_embed/{n}.pt")[0:1].to(dev,dt)
feats={"audio_emb":aemb,"local_audio_emb":law,"text_emb":te,"total_frames":T}
ar_mo=generate_ar_motion(feats,ar,dev,dt,1.5,1.0,seed_frame=seed,use_lcm=True,temperature=1.0)  # [T,512]
ar_mo=ar_mo[:F_].reshape(1,F_,32,16).to(dev,dt)
gt_mo=load_motion_gt(f"{ROOT}/pose_embed/{n}.pt")[:F_].reshape(1,F_,32,16).to(dev,dt)
print(f"[motion] AR std={ar_mo.float().std():.3f} GT std={gt_mo.float().std():.3f}", flush=True)

@torch.no_grad()
def render_save(motion, tag):
    M.set_reference(ref_lat, clip, 1)
    noise=torch.randn(1,4,F_,64,64,device=dev,dtype=dt)
    x0,_=self_forcing_rollout(M,noise,clip,motion,[999,749,499,249],block_size=8,grad_window=None,full_steps=True)
    z=rearrange(x0,"b c f h w -> (b f) c h w")/0.18215
    for f in [0,24,48,63]:
        img=(vae.decode(z[f:f+1],1).sample/2+0.5).clamp(0,1)[0]
        Image.fromarray((img.permute(1,2,0).cpu().float().numpy()*255).astype("uint8")).save(f"{OUT}/arc_{tag}_f{f}.png")
    print(f"[{tag}] x0.std={x0.float().std():.3f}", flush=True)

render_save(gt_mo, "gt")
render_save(ar_mo, "ar")
print("DONE", flush=True)

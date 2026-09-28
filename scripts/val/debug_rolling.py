"""rolling KV-cache(无 reset) + 高 CFG 对比。M023 全长, self_forcing_rollout 单次 + set_window(W)。
每个配置: 存关键帧 + 逐帧差(看 63->64 跳变是否消失) + 平均帧差(动态 proxy)。CUDA_VISIBLE_DEVICES 指定。"""
import os, sys, torch, numpy as np
os.environ["AR_REPO_ROOT"]="/media/ps/ssd5/ayr/motar"; os.environ["XNEMO_REPO_ROOT"]="/media/ps/ssd5/ayr/x-nemo-inference"
for p in ("/media/ps/ssd5/ayr/motar","/media/ps/ssd5/ayr/x-nemo-inference","/media/ps/ssd5/ayr/x-nemo-inference/scripts/val"):
    sys.path.append(p)
import librosa
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.rollout import self_forcing_rollout
from test_ar_model import (load_encoders, load_ar_model, generate_ar_motion, load_motion_gt,
                           normalize_wav, align_to_frames, build_local_window, AUDIO_SR, FPS, WINDOW_SIZE)
dev=torch.device("cuda:0"); dt=torch.bfloat16
CKPT="/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
XCFG="/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml"
AR_CKPT=("/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD_video_v3/"
         "20260702_080807_sf_gan_v32/checkpoints/sf_gan_step_3000.pt")
ROOT="/media/ps/ssd5/ayr/MEAD_frames_512_25fps"; n="M023_video_right_30_fear_level_3_010"
OUT="/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"
WIN=48
cfg=OmegaConf.load(XCFG)
M=DMD2Models(dev,gen_ckpt=None,block_size=8); rk=torch.load(CKPT,map_location="cpu")
M.generator.load_state_dict(rk["generator_ema"],strict=False); M.generator.eval()
vae=AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev,dt).eval()
ar=load_ar_model(cfg,AR_CKPT,dev,dt,1); tok,txte,aproc,aenc=load_encoders(cfg,dev)
print("[loaded]",flush=True)
ref_pil=Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512,512))
clip=M.image_encoder(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)),return_tensors="pt").pixel_values.to(dev,dt)).image_embeds.unsqueeze(1)
ref_lat=torch.load(f"{ROOT}/frame_latent/{n}.pt",map_location="cpu").float()[0:1].to(dev,dt)
audio,_=librosa.load(f"{ROOT}/audio/{n}.wav",sr=AUDIO_SR); T=len(audio)//(AUDIO_SR//FPS); audio=audio[:T*(AUDIO_SR//FPS)]
ac=torch.from_numpy(audio).float().unsqueeze(0).to(dev)
with torch.no_grad():
    aemb=aenc(normalize_wav(ac)).last_hidden_state; ff=align_to_frames(aemb,T); law=build_local_window(ff,half=WINDOW_SIZE//2)
    cp=f"{ROOT}/emo_pose_caption/{n}.txt"; cap=open(cp).read().strip() if os.path.exists(cp) else "a person is talking"
    tk=tok(cap,padding="max_length",truncation=True,max_length=128,return_tensors="pt"); ti,tm=tk.input_ids.to(dev),tk.attention_mask.to(dev)
    with torch.cuda.amp.autocast(dtype=torch.float32): te=txte(input_ids=ti,attention_mask=tm).last_hidden_state
    te=te*tm.unsqueeze(-1)
seed=load_motion_gt(f"{ROOT}/pose_embed/{n}.pt")[0:1].to(dev,dt)
feats={"audio_emb":aemb,"local_audio_emb":law,"text_emb":te,"total_frames":T}

@torch.no_grad()
def decode(x0):
    z=rearrange(x0,"b c f h w -> (b f) c h w")/0.18215
    outs=[vae.decode(z[i:i+2],z[i:i+2].shape[0]).sample for i in range(0,z.shape[0],2)]
    v=(torch.cat(outs,0)/2+0.5).clamp(0,1)
    return (v.permute(0,2,3,1).cpu().float().numpy()*255).astype(np.uint8)  # [F,H,W,3]

@torch.no_grad()
def run(cfg_a, tag):
    mo=generate_ar_motion(feats,ar,dev,dt,cfg_a,1.0,seed_frame=seed,use_lcm=True,temperature=1.0)
    Tm=mo.shape[0]; Tm-=Tm%8; mo=mo[:Tm].reshape(1,Tm,32,16).to(dev,dt)
    M.set_reference(ref_lat,clip,1)
    M.gen_causal.set_window(WIN)  # ★ rolling window, 无 reset, 单次全长 rollout
    noise=torch.randn(1,4,Tm,64,64,device=dev,dtype=dt)
    x0,_=self_forcing_rollout(M,noise,clip,mo,[999,749,499,249],block_size=8,grad_window=None,full_steps=True)
    arr=decode(x0)  # [T,H,W,3]
    for f in [0,40,63,64,80,128,min(155,Tm-1)]:
        Image.fromarray(arr[f]).save(f"{OUT}/roll_{tag}_f{f}.png")
    g=arr.astype(np.float32).mean(3)  # gray
    d=np.abs(np.diff(g,axis=0)).mean(axis=(1,2))
    j64=float(d[63]) if len(d)>63 else -1; j128=float(d[127]) if len(d)>127 else -1
    print(f"[{tag}] cfg_a={cfg_a} motion_std={mo.float().std():.3f} x0.std={x0.float().std():.3f} "
          f"T={Tm} meandiff={d.mean():.2f} jump63->64={j64:.2f} jump127->128={j128:.2f}", flush=True)

run(1.5,"cfg15")
run(2.5,"cfg25")
print("DONE",flush=True)

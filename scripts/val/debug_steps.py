"""步数→动态 验证: ODE-baseline(蒸馏前)8步 vs dmd2学生 4步/8步。GT motion, rolling win48。
逐帧差=动态 proxy。CUDA_VISIBLE_DEVICES 指定。"""
import sys, torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.rollout import self_forcing_rollout
dev=torch.device("cuda:0"); dt=torch.bfloat16
ODE="/media/ps/ssd5/ayr/x-nemo-inference/output/ode_init/ode_step_8000.pt"
DMD2="/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
ROOT="/media/ps/ssd5/ayr/MEAD_frames_512_25fps"; n="M023_video_right_30_fear_level_3_010"; F_=160; WIN=48
OUT="/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"
DSL4=[999,749,499,249]; DSL8=[999,874,749,624,499,374,249,124]
cfg=OmegaConf.load(DEC_CFG)
M=DMD2Models(dev,gen_ckpt=ODE,block_size=8)   # generator = ODE baseline(蒸馏前多步)
vae=AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev,dt).eval()
ref_pil=Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512,512))
clip=M.image_encoder(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)),return_tensors="pt").pixel_values.to(dev,dt)).image_embeds.unsqueeze(1)
ref_lat=torch.load(f"{ROOT}/frame_latent/{n}.pt",map_location="cpu").float()[0:1].to(dev,dt)
mo=torch.load(f"{ROOT}/pose_embed/{n}.pt",map_location="cpu").float().reshape(-1,32*16)[:F_].reshape(1,F_,32,16).to(dev,dt)
print("[loaded]",flush=True)

@torch.no_grad()
def render(dsl,tag):
    M.set_reference(ref_lat,clip,1); M.gen_causal.set_window(WIN)
    noise=torch.randn(1,4,F_,64,64,device=dev,dtype=dt)
    x0,_=self_forcing_rollout(M,noise,clip,mo,dsl,block_size=8,grad_window=None,full_steps=True)
    z=rearrange(x0,"b c f h w -> (b f) c h w")/0.18215
    outs=[vae.decode(z[i:i+2],z[i:i+2].shape[0]).sample for i in range(0,z.shape[0],2)]
    v=(torch.cat(outs,0)/2+0.5).clamp(0,1)
    arr=(v.permute(0,2,3,1).cpu().float().numpy()*255).astype(np.uint8)
    Image.fromarray(arr[40]).save(f"{OUT}/steps_{tag}_f40.png")
    g=arr.astype(np.float32).mean(3); d=np.abs(np.diff(g,axis=0)).mean()
    print(f"[{tag}] steps={len(dsl)} x0.std={x0.float().std():.3f} meandiff(动态)={d:.3f}",flush=True)

render(DSL8,"ode8")   # ODE baseline 8步(参照:蒸馏前多步动态)
# 换 dmd2 学生 EMA
rk=torch.load(DMD2,map_location="cpu"); M.generator.load_state_dict(rk["generator_ema"],strict=False)
print("[loaded dmd2 ema]",flush=True)
render(DSL4,"dmd2_4")  # 当前部署: 学生 4步
render(DSL8,"dmd2_8")  # 学生 8步(off-distribution)
print("DONE",flush=True)

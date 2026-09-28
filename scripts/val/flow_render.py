"""flow-teacher 采样器: rectified-flow Euler ODE(双向 teacher),验证少步采样。
z_t=(1-t)x0+t·ε; 从 t=1(噪声)Euler 积分到 t=0(数据): z ← z - (1/N)·v_θ(z, t·999)。
用法: CUDA_VISIBLE_DEVICES=N python scripts/val/flow_render.py --ckpt output/flow_teacher/flow_step_XXXX.pt --steps 4 8"""
import sys, os, argparse, torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT = "/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sample", default="M023_video_right_30_fear_level_3_010")
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--steps", type=int, nargs="+", default=[4, 8])
args = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
# ★ flow 权重覆盖
rk = torch.load(args.ckpt, map_location="cpu")
denu.load_state_dict(rk["denoising_unet"], strict=False)
denu.eval()
print(f"[loaded flow ckpt {args.ckpt} step={rk.get('step')} obj={rk.get('objective')}]", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()

n = args.sample; F_ = args.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
ref_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[0:1].to(dev, dt)
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)

rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i + 2], z[i:i + 2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def sample(N, shift=1.0):
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    sched.set_timesteps(N, device=dev)
    z = torch.randn(1, 4, F_, 64, 64, device=dev, dtype=dt)          # sigma=1 纯噪声
    for t in sched.timesteps:
        v = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
        z = sched.step(v.float(), t, z.float()).prev_sample.to(dt)   # FlowMatch Euler
    arr = decode(z)
    for f in [0, 40, min(63, F_ - 1)]:
        Image.fromarray(arr[f]).save(f"{OUT}/flow_N{N}_f{f}.png")
    g = arr.astype(np.float32).mean(3); d = np.abs(np.diff(g, axis=0)).mean()
    print(f"[flow N={N}] x0.std={z.float().std():.3f} frame-diff={d:.3f} 帧存 flow_N{N}_f*.png", flush=True)


for N in args.steps:
    sample(N)
print("DONE", flush=True)

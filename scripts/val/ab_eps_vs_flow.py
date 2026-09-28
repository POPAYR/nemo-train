"""决定性 A/B:原始 ε-DDPM teacher vs flow teacher —— 同样本、同渲染方式、同步数、都不开 CFG。
回答「闪烁/伪影是 flow 参数化不适配，还是 flow 微调训练不足」。
两者唯一差异 = 参数化(ε-pred + DDIM  vs  v-pred + FlowMatchEuler)与是否经过 flow 微调。
指标:一阶差分(运动)、二阶差分(抖动)、抖动比(=二阶/一阶,归一化掉运动幅度)、空间高频(锐度)。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/ab_eps_vs_flow.py --steps 35
"""
import sys, os, argparse, subprocess
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope, collect_temporal_self_attns

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
FLOW = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_teacher/flow_teacher_FINAL.pt"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_flow/_ab"

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="M003_video_front_happy_level_3_001")
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--flow_ckpt", default=FLOW, help="flow teacher ckpt;是否用 RoPE 由 ckpt 内 'rope' 标记自动识别")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
os.makedirs(OUT, exist_ok=True)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
_SP = torch.load(cfg.denoising_unet_path, map_location="cpu")
_TP = torch.load(cfg.temporal_module_path, map_location="cpu")
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

n = a.sample; F_ = a.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
_all = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()
ref_lat = _all[0:1].to(dev, dt)
gt_lat = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(g[2:] - 2*g[1:-1] + g[:-2]).mean()
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()
    return d1, d2, d2/d1, hf


def setref():
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)


def noise0():
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    return torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)


@torch.no_grad()
def run_eps(N):
    """原始 ε-DDPM teacher + DDIM。"""
    denu.load_state_dict(_SP, strict=False); denu.load_state_dict(_TP, strict=False); denu.eval()
    set_temporal_rope(denu, False, mode="off")     # ε teacher 用原始加性 PE
    setref()
    sch = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
    sch.set_timesteps(N, device=dev)
    z = noise0() * sch.init_noise_sigma
    for t in sch.timesteps:
        eps = denu(z, t, encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
        z = sch.step(eps.float(), t, z.float()).prev_sample.to(dt)
    return z


@torch.no_grad()
def run_flow(N):
    """flow teacher + FlowMatchEuler。"""
    denu.load_state_dict(_SP, strict=False); denu.load_state_dict(_TP, strict=False)
    _rk = torch.load(a.flow_ckpt, map_location="cpu")
    denu.load_state_dict(_rk["denoising_unet"], strict=False); denu.eval()
    # ★ 位置编码必须与训练时一致,否则静默错配:从 ckpt 的 rope 标记自动识别
    _use_rope = bool(_rk.get("rope", False))
    set_temporal_rope(denu, _use_rope, mode="bidir" if _use_rope else "off")
    if not hasattr(run_flow, "_logged"):
        print(f"    (flow ckpt step={_rk.get('step')} rope={_use_rope} → "
              f"{'RoPE/bidir' if _use_rope else '加性PE/原processor'})", flush=True)
        run_flow._logged = True
    setref()
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    sch.set_timesteps(N, device=dev)
    z = noise0()
    for t in sch.timesteps:
        v = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
        z = sch.step(v.float(), t, z.float()).prev_sample.to(dt)
    return z


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


print(f"\n样本 {n}  {F_}帧一次性渲染  N={a.steps}步  无CFG  seed={a.seed}")
print(f"{'模型':>22} | {'一阶(运动)':>10} {'二阶(抖动)':>10} {'抖动比':>8} {'空间高频':>9} {'x0.std':>7}")
print("-" * 78)
ga = decode(gt_lat); d1, d2, r, hf = metrics(ga)
print(f"{'GT':>22} | {d1:>10.3f} {d2:>10.3f} {r:>8.3f} {hf:>9.2f} {gt_lat.float().std():>7.3f}")

for tag, fn in [("原始ε-DDPM + DDIM", run_eps), ("flow(v-pred) + Euler", run_flow)]:
    x0 = fn(a.steps)
    arr = decode(x0); d1, d2, r, hf = metrics(arr)
    print(f"{tag:>22} | {d1:>10.3f} {d2:>10.3f} {r:>8.3f} {hf:>9.2f} {x0.float().std():>7.3f}")
    save_mp4(arr, f"{OUT}/{n}_{'eps' if 'ε' in tag else 'flow'}_N{a.steps}.mp4")

print("\n判读:抖动比(二阶/一阶)已归一化运动幅度。若 ε 的抖动比也远高于 GT → 与 flow 无关(渲染方式/模型固有);")
print("      若 ε 接近 GT 而 flow 明显更高 → flow 微调确实劣化了时序一致性。")
print("DONE", flush=True)

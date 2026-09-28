"""flow teacher 推理超参扫描:步数 N × CFG × shift，找甜点。
⭐ CFG 负分支复刻原 XNeMo pipeline 的语义:neg = **参考帧自己的 motion**（"保持参考图不动"），
   即 mo[:, 0:1].expand_as(mo)（原实现见 pipeline_pose2vid_motenc_long.py:522 neg_motion_hidden_states）。
   旧 DDPM teacher 评测/showcase 用的是 steps=35 + cfg=2.5，我们的 flow 渲染此前一直 cfg=1（=关）。
同时打印 GT 的 frame-diff 作为「闪烁」的参照基准——高于 GT 才叫多余闪烁。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/sweep_flow_teacher.py --mode steps
      CUDA_VISIBLE_DEVICES=0 python scripts/val/sweep_flow_teacher.py --mode cfg --steps 35
"""
import sys, os, argparse, subprocess
import torch, numpy as np
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
TEACHER = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_teacher/flow_teacher_FINAL.pt"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_flow/_sweep"

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="M003_video_front_happy_level_3_001")
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--mode", choices=["steps", "cfg", "shift"], default="steps")
ap.add_argument("--steps", type=int, default=35, help="mode=cfg/shift 时固定的步数")
ap.add_argument("--cfg", type=float, default=1.0, help="mode=steps/shift 时固定的 CFG")
ap.add_argument("--shift", type=float, default=1.0)
ap.add_argument("--save_video", action="store_true")
ap.add_argument("--ckpt", default=TEACHER, help="flow ckpt;是否 RoPE 由 ckpt 内标记自动识别")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfgm = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfgm.inference_config)
os.makedirs(OUT, exist_ok=True)

refu = UNet2DConditionModel.from_pretrained(cfgm.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfgm.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfgm.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfgm.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfgm.temporal_module_path, map_location="cpu"), strict=False)
_rk = torch.load(a.ckpt, map_location="cpu")
denu.load_state_dict(_rk["denoising_unet"], strict=False)
denu.eval()
_use_rope = bool(_rk.get("rope", False))
if _use_rope:
    from src.models.temporal_causal import set_temporal_rope
    set_temporal_rope(denu, True, mode="bidir")
print(f"[ckpt] {a.ckpt}  step={_rk.get('step')} stage={_rk.get('stage')} rope={_use_rope}", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfgm.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfgm.vae_path).to(dev, dt).eval()
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

n = a.sample; F_ = a.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
_all = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()
ref_lat = _all[0:1].to(dev, dt)
gt_lat = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
mo_neg = mo[:, 0:1].expand_as(mo).contiguous()      # ★ 负分支 = 参考帧(首帧)的 motion，铺满全序列


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    fd = np.abs(np.diff(g, axis=0)).mean()                        # 相邻帧差(闪烁/动态)
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()   # 空间高频(锐度代理)
    return fd, hf


@torch.no_grad()
def sample(N, cfg_s, shift):
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    sched.set_timesteps(N, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    z = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)
    for t in sched.timesteps:
        te = t.expand(1).to(dt)
        v_c = denu(z, te, encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
        if cfg_s > 1.0:
            v_u = denu(z, te, encoder_hidden_states=[clip, mo_neg], pose_cond_fea=None, return_dict=False)[0]
            v = v_u.float() + cfg_s * (v_c.float() - v_u.float())
        else:
            v = v_c.float()
        z = sched.step(v, t, z.float()).prev_sample.to(dt)
    return z


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


# ---- GT 基准 ----
gt_arr = decode(gt_lat)
gfd, ghf = metrics(gt_arr)
print(f"\n样本 {n}  {F_}帧  seed={a.seed}")
print(f"[GT 基准]  x0.std={gt_lat.float().std():.3f}  frame-diff={gfd:.3f}  空间高频={ghf:.2f}")
print("  ↑ frame-diff 高于 GT 才算「多余闪烁」;低于 GT 说明动态被压平\n")

if a.mode == "steps":
    grid = [(N, a.cfg, a.shift) for N in [4, 8, 16, 25, 35, 50]]
    print(f"{'N':>4} {'cfg':>5} | {'x0.std':>7} {'frame-diff':>10} {'Δvs GT':>8} {'空间高频':>9}")
elif a.mode == "cfg":
    grid = [(a.steps, c, a.shift) for c in [1.0, 1.5, 2.0, 2.5, 3.0]]
    print(f"{'N':>4} {'cfg':>5} | {'x0.std':>7} {'frame-diff':>10} {'Δvs GT':>8} {'空间高频':>9}")
else:
    grid = [(a.steps, a.cfg, s) for s in [1.0, 2.0, 3.0, 5.0]]
    print(f"{'N':>4} {'shift':>5} | {'x0.std':>7} {'frame-diff':>10} {'Δvs GT':>8} {'空间高频':>9}")
print("-" * 56)

for N, cs, sh in grid:
    x0 = sample(N, cs, sh)
    arr = decode(x0)
    fd, hf = metrics(arr)
    key = sh if a.mode == "shift" else cs
    print(f"{N:>4} {key:>5.1f} | {x0.float().std():>7.3f} {fd:>10.3f} {fd-gfd:>+8.3f} {hf:>9.2f}", flush=True)
    if a.save_video:
        save_mp4(arr, f"{OUT}/{n}_N{N}_cfg{cs}_shift{sh}.mp4")

print("\nDONE", flush=True)

"""三阶段可视化：flow teacher(双向) / 因果 init(TF) / flow-DMD student —— 同样本同噪声，直接可比。
输出 mp4 + 帧条 png 到 output/viz_flow/<sample>/。
用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/val/viz_flow_stages.py \
      --samples M023_video_right_30_fear_level_3_010 M003_video_front_happy_level_3_001 --frames 64
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
from src.models.temporal_causal import TemporalCausalControl
from src.distill.flow_math import flow_add_noise, v_to_x0, flow_step_list

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_flow"

# 三个阶段(名字 → ckpt, 模式, 步数)
STAGES = [
    ("1_teacher_N35",  "output/flow_teacher/flow_teacher_FINAL.pt",          "bidir",  35),
    ("1_teacher_N4",   "output/flow_teacher/flow_teacher_FINAL.pt",          "bidir",   4),
    ("2_causal_init",  "output/ode_init_flow_tf/flow_tf_step_500.pt",        "causal",  4),
    ("3_dmd_v3",       "output/flowdmd_v3_sigcap/ckpt/dmd2_step_500.pt",     "causal",  4),
]

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", default=["M023_video_right_30_fear_level_3_010"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--gt", action="store_true", help="额外导出 GT 视频(由 GT latent 解码)作为参照")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

# ---- 共享组件(只建一次) ----
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
_BASE_SPATIAL = torch.load(cfg.denoising_unet_path, map_location="cpu")
_BASE_TEMPORAL = torch.load(cfg.temporal_module_path, map_location="cpu")
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
ctrl = TemporalCausalControl(denu, block_size=a.block, window=0)
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")


def load_stage(ckpt):
    """把 ckpt 权重灌进共享 denu(先恢复基座，再覆盖)。"""
    denu.load_state_dict(_BASE_SPATIAL, strict=False)
    denu.load_state_dict(_BASE_TEMPORAL, strict=False)
    rk = torch.load(ckpt, map_location="cpu")
    sd = rk.get("denoising_unet") or rk.get("generator_ema") or rk.get("generator")
    key = "denoising_unet" if "denoising_unet" in rk else ("generator_ema" if rk.get("generator_ema") else "generator")
    denu.load_state_dict(sd, strict=False)
    denu.eval()
    return rk.get("step"), key


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i + 2], z[i:i + 2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def save_mp4(arr, path, fps):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


def save_strip(arr, path, n=8):
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    H, W = arr.shape[1], arr.shape[2]
    row = Image.new("RGB", (W * len(idx), H))
    for j, i in enumerate(idx):
        row.paste(Image.fromarray(arr[i]), (W * j, 0))
    row.save(path)
    return idx.tolist()


@torch.no_grad()
def run_bidir(N, clip, mo, ref_lat, noise):
    """双向 teacher:整段 FlowMatchEuler 采样。"""
    ctrl.set_mode("off"); ctrl.reset_cache()
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    sched.set_timesteps(N, device=dev)
    z = noise.clone()
    for t in sched.timesteps:
        v = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
        z = sched.step(v.float(), t, z.float()).prev_sample.to(dt)
    return z


@torch.no_grad()
def run_causal(N, clip, mo, ref_lat, noise):
    """因果流式:逐 block N 步 Euler + 自预测干净 commit(= SF 推理)。"""
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    _ts, _sg = flow_step_list(N, shift=1.0)
    sigma_list = [float(x) for x in _sg[:-1].tolist()]
    ctrl.set_mode("stream"); ctrl.reset_cache()
    F_ = noise.shape[2]; BL = a.block
    outs = []

    def denoise(zz, sig, mb):
        te = torch.full((zz.shape[0],), float(sig) * 1000.0, device=dev, dtype=dt)
        v = denu(zz, te, encoder_hidden_states=[clip, mb], pose_cond_fea=None, return_dict=False)[0]
        return v_to_x0(zz, v, torch.tensor(float(sig), device=dev))

    for cur in range(0, F_, BL):
        z = noise[:, :, cur:cur + BL]; mb = mo[:, cur:cur + BL]
        ctrl.set_offset(cur)
        x0 = None
        for i, sig in enumerate(sigma_list):
            ctrl.set_commit(False)
            x0 = denoise(z, sig, mb)
            if i == len(sigma_list) - 1: break
            z = flow_add_noise(x0, torch.randn_like(x0), sigma_list[i + 1]).to(dt)
        outs.append(x0)
        ctrl.set_commit(True)
        denoise(x0, 0.0, mb)
    ctrl.set_commit(True); ctrl.set_mode("off")
    return torch.cat(outs, dim=2)


for name in a.samples:
    F_ = a.frames
    od = os.path.join(OUT_ROOT, name); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    ref_pil.save(f"{od}/ref.png")
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    noise = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)   # ★ 各阶段共用同一噪声
    print(f"\n===== {name}  ({F_} 帧, seed={a.seed}) =====", flush=True)

    if a.gt:
        gt = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
        arr = decode(gt)
        save_mp4(arr, f"{od}/0_GT.mp4", a.fps); save_strip(arr, f"{od}/0_GT_strip.png")
        print(f"  [GT]            std={gt.float().std():.3f}  → 0_GT.mp4", flush=True)

    for tag, ckpt, mode, N in STAGES:
        if not os.path.exists(ckpt):
            print(f"  [skip] {tag}: 缺 {ckpt}", flush=True); continue
        step, key = load_stage(ckpt)
        x0 = run_bidir(N, clip, mo, ref_lat, noise) if mode == "bidir" else run_causal(N, clip, mo, ref_lat, noise)
        arr = decode(x0)
        save_mp4(arr, f"{od}/{tag}.mp4", a.fps)
        marks = save_strip(arr, f"{od}/{tag}_strip.png")
        gray = arr.astype(np.float32).mean(3); fd = np.abs(np.diff(gray, axis=0)).mean()
        print(f"  [{tag:16s}] {mode:6s} N={N:<2d} step={step} key={key}  "
              f"x0.std={x0.float().std():.3f}  frame-diff={fd:.3f}  → {tag}.mp4 / _strip.png(帧{marks})", flush=True)

print("\nDONE", flush=True)

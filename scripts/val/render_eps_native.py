"""用 ε teacher 的**正规配置**渲染:24帧滑窗(overlap 4) + CFG 2.5 + DDIM 35步。
这是评测/showcase 一直用的配置(见 eval_metrics/showcase_selby.py 与 RESULTS_ours.md 的 renderer 上界),
也是 flow teacher 真正要超越的质量靶子 —— 之前 A/B 里的 ε 是被剥掉 CFG 和滑窗的「降级版」。
输出 mp4 + 同一套指标(一阶/二阶/抖动比/空间高频),可直接与 ab_eps_vs_flow.py 的数字对齐。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/render_eps_native.py --frames 64 --cfg 2.5
"""
import sys, os, argparse, subprocess
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference/scripts/val")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_flow/_ab"

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="M003_video_front_happy_level_3_001")
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--context_frames", type=int, default=24)
ap.add_argument("--context_overlap", type=int, default=4)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
dev = torch.device("cuda:0"); os.makedirs(OUT, exist_ok=True)
cfg = OmegaConf.load(DEC_CFG)

from test_ar_model import load_xnemo_pipeline, render
pipe = load_xnemo_pipeline(cfg, dev, torch.float16)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, torch.bfloat16).eval()

n = a.sample; F_ = a.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32 * 16)[:F_]
_all = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()
gt_lat = _all[:F_].to(dev, torch.bfloat16).permute(1, 0, 2, 3).unsqueeze(0)


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean()
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()
    return d1, d2, d2 / d1, hf


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


print(f"\n样本 {n}  {F_}帧  ε teacher 正规配置:{a.steps}步 DDIM / CFG={a.cfg} / "
      f"{a.context_frames}帧滑窗(overlap {a.context_overlap})")
print(f"{'配置':>26} | {'一阶(运动)':>10} {'二阶(抖动)':>10} {'抖动比':>8} {'空间高频':>9}")
print("-" * 72)
ga = decode(gt_lat); d1, d2, r, hf = metrics(ga)
print(f"{'GT':>26} | {d1:>10.3f} {d2:>10.3f} {r:>8.3f} {hf:>9.2f}")

g = torch.Generator(device=dev); g.manual_seed(a.seed)
args_ns = argparse.Namespace(W=512, H=512, steps=a.steps, cfg=a.cfg,
                             context_frames=a.context_frames, context_overlap=a.context_overlap)
vid = render(pipe, ref_pil, ref_pil, mo.unsqueeze(0).to(dev), args_ns, g)   # [1,3,T,H,W] in [0,1]
arr = (vid[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
d1, d2, r, hf = metrics(arr)
tag = f"ε正规(cfg{a.cfg},窗{a.context_frames})"
print(f"{tag:>26} | {d1:>10.3f} {d2:>10.3f} {r:>8.3f} {hf:>9.2f}")
path = f"{OUT}/{n}_eps_NATIVE_cfg{a.cfg}_win{a.context_frames}.mp4"
save_mp4(arr, path)
print(f"\n→ {path}")
print("DONE", flush=True)

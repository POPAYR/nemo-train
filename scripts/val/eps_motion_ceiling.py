"""探针:motion latent 到底能驱动多大运动?—— 用原版 ε XNeMo(官方权重)做上界。

动机:我们的 flow 模型在所有 ckpt/所有 cfg 下都比 GT 欠运动 15-20%。
候选解释 (a) 我们 stage1+stage2 流水线丢了信息;(b) motion latent 本身信息不够。
原版 ε 是 motion latent 的"原生消费者"(官方就是拿它训的),所以它是这套条件表示的上界探针。

必须关掉滑窗:24帧窗/overlap4 的重叠平均本身就是强力方差抑制,会压低运动,
把"滑窗压的"和"latent 不够"混在一起。context_frames >= frames 且 overlap=0 即退化为单次。
本脚本同时跑单次与滑窗两种,把滑窗的贡献单独剥出来。

指标与 render_cfg_compare.py 完全一致。输出 → output/eval/eps_ceiling/
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
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/eps_ceiling"

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
dev = torch.device("cuda:0"); os.makedirs(OUT, exist_ok=True)
cfg = OmegaConf.load(DEC_CFG)

from test_ar_model import load_xnemo_pipeline, render
pipe = load_xnemo_pipeline(cfg, dev, torch.float16)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, torch.bfloat16).eval()


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


F_ = a.frames
CONFIGS = [("单次", F_, 0), ("滑窗24ov4", 24, 4)]
agg = {k: [] for k, _, _ in CONFIGS}; agg["GT"] = []

for n in a.samples:
    od = os.path.join(OUT, n); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
    mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_]
    gt_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[:F_] \
             .to(dev, torch.bfloat16).permute(1, 0, 2, 3).unsqueeze(0)
    print(f"\n===== {n}  {F_}帧 =====", flush=True)
    m = metrics(decode(gt_lat)); agg["GT"].append(m)
    print(f"{'GT(VAE往返)':>14} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)
    for tag, cf, ov in CONFIGS:
        g = torch.Generator(device=dev); g.manual_seed(a.seed)
        ns = argparse.Namespace(W=512, H=512, steps=a.steps, cfg=a.cfg,
                                context_frames=cf, context_overlap=ov)
        vid = render(pipe, ref_pil, ref_pil, mo.unsqueeze(0).to(dev), ns, g)
        arr = (vid[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
        m = metrics(arr); agg[tag].append(m)
        save_mp4(arr, f"{od}/eps_{tag}_N{a.steps}_cfg{a.cfg}_{F_}f.mp4")
        print(f"{'ε '+tag:>14} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)

print("\n" + "=" * 62)
print(f"N={len(a.samples)} 均值 | {'运动':>7} {'抖动':>7} {'抖动比':>7} {'高频':>6}")
gt = np.array(agg["GT"]).mean(0)
print(f"{'GT(VAE往返)':>14} | {gt[0]:>7.3f} {gt[1]:>7.3f} {gt[2]:>7.3f} {gt[3]:>6.2f}")
for tag, _, _ in CONFIGS:
    v = np.array(agg[tag]).mean(0)
    print(f"{'ε '+tag:>14} | {v[0]:>7.3f} {v[1]:>7.3f} {v[2]:>7.3f} {v[3]:>6.2f}"
          f"   运动相对GT {(v[0]-gt[0])/gt[0]*100:>+6.1f}%")
print("DONE", flush=True)

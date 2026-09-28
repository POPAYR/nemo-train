"""损失阶梯第 1-2 级:原始 JPEG 帧 vs VAE 往返重建。
动机:我们一直把 "GT" 定义为 frame_latent 过 VAE 解码的结果,但那已经损失了一次。
若 VAE 往返本身就压掉了运动/高频,则"欠运动"里有一部分根本不是模型的锅。
指标与 render_cfg_compare.py 的 metrics() 完全一致(运动/抖动/抖动比/高频)。
输出 → output/eval/vae_ladder/
"""
import sys, os, argparse
import numpy as np, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/vae_ladder"

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--frames", type=int, default=40)
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean()
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()
    return d1, d2, d2 / d1, hf


@torch.no_grad()
def decode(lat):
    z = lat.to(dev, dt) / 0.18215
    outs = [vae.decode(z[i:i + 2], z[i:i + 2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


print(f"{'sample':<42}{'级别':<10}{'运动':>8}{'抖动':>8}{'抖动比':>8}{'高频':>8}")
print("-" * 90)
agg = {"raw": [], "vae": []}
for name in a.samples:
    fs = sorted(os.listdir(f"{ROOT}/face_frames/{name}"))[:a.frames]
    raw = np.stack([np.array(Image.open(f"{ROOT}/face_frames/{name}/{f}").convert("RGB").resize((512, 512)))
                    for f in fs])
    lat = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()[:a.frames]
    rec = decode(lat)
    for tag, arr in [("raw原始帧", raw), ("VAE往返", rec)]:
        m = metrics(arr)
        agg["raw" if tag.startswith("raw") else "vae"].append(m)
        print(f"{name[:40]:<42}{tag:<10}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>8.3f}")

print("-" * 90)
r = np.array(agg["raw"]).mean(0); v = np.array(agg["vae"]).mean(0)
print(f"{'均值':<42}{'raw原始帧':<10}{r[0]:>8.3f}{r[1]:>8.3f}{r[2]:>8.3f}{r[3]:>8.3f}")
print(f"{'均值':<42}{'VAE往返':<10}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}{v[3]:>8.3f}")
print(f"{'VAE往返相对原始帧的变化(%)':<42}{'':<10}" +
      "".join(f"{(v[i]-r[i])/r[i]*100:>+8.1f}" for i in range(4)))
print("DONE")

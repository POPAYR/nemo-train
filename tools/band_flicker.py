import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys as _sys
if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
#!/usr/bin/env python3
"""量样本视频的分区域帧间闪烁,并与 GT / VAE 地板对照。

★ 样本 mp4 是 **1024x512 左右拼接**(左 GT 右生成),必须按宽度对半切 ——
  整幅丢进去会得到 60x 这种荒谬数字(DATA.md §17 记的坑)。
★ 掩码统一用 GT 半幅算,再套到生成半幅上:两边裁剪同源,区域可比;
  用生成半幅自己算掩码会让边界随生成内容抖动。
"""
import argparse, glob, os
import cv2, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--names", nargs="*", default=None)
ap.add_argument("--vae_floor", type=int, default=1, help="是否同时算 VAE 往返地板")
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()

import mediapipe as mp
seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
if a.vae_floor:
    from diffusers import AutoencoderKLTemporalDecoder
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        XP("XN_PRETRAINED", "stable-video-diffusion-img2vid/vae")).to("cuda", torch.float16).eval()


def halves(p, n):
    cap = cv2.VideoCapture(p); L, R = [], []
    while len(L) < n:
        ok, f = cap.read()
        if not ok: break
        w = f.shape[1] // 2
        L.append(f[:, :w]); R.append(f[:, w:])
    cap.release(); return L, R


def masks(bgr, stride=2):
    acc = None
    for i, f in enumerate(bgr):
        if i % stride: continue
        im = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
        b = seg.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).segmentation_mask > 0.5
        acc = b if acc is None else (acc | b)
    u = acc.astype(np.uint8); wide = cv2.dilate(u, K(45)) > 0; near = cv2.dilate(u, K(8)) > 0
    inner = cv2.erode(u, K(12)) > 0
    # ★ 发丝 = 轮廓环(dilate12 \ erode12),DATA.md §23。旧"边界带"混了纯背景,仅保留作对照
    return {"发丝": (cv2.dilate(u, K(12)) > 0) & ~inner, "人物内部": inner, "背景": ~wide,
            "边界带(旧)": wide & ~near}


def bd1(bgr, m):
    g = [cv2.cvtColor(cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.float32) for f in bgr]
    d = [np.abs(g[i] - g[i - 1]) for i in range(1, len(g))]
    return {k: float(np.median([x[v].mean() for x in d])) for k, v in m.items()}


def vae_rt(gt):
    rec = []
    with torch.no_grad():
        for i in range(0, len(gt), 8):
            x = np.stack([cv2.cvtColor(cv2.resize(f, (512, 512)), cv2.COLOR_BGR2RGB) for f in gt[i:i + 8]])
            t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(127.5).sub(1).to("cuda", torch.float16)
            z = vae.encode(t).latent_dist.mean * 0.18215          # 与 tool/encode_latents.py 同口径
            y = vae.decode(z / 0.18215, z.shape[0]).sample
            y = y.add(1).mul(127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            rec += [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in y]
    return rec


names = a.names or [os.path.basename(d.rstrip("/")) for d in a.dirs]
agg = {}
for d, nm in zip(a.dirs, names):
    for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
        gt, gen = halves(p, a.n)
        if len(gt) < 8: continue
        m = masks(gt)
        g, q = bd1(gt, m), bd1(gen, m)
        agg.setdefault(("GT", "-"), []).append(g)
        agg.setdefault((nm, "生成"), []).append(q)
        if a.vae_floor:
            agg.setdefault(("VAE往返", "地板"), []).append(bd1(vae_rt(gt), m))

_cols = ["发丝", "人物内部", "背景", "边界带(旧)"]
print("%-26s" % "来源" + "".join("%12s" % c for c in _cols))
for (nm, tag), rows in agg.items():
    print("%-26s" % (nm + ("" if tag == "-" else " " + tag)) +
          "".join("%12.3f" % np.median([r[c] for r in rows]) for c in _cols))

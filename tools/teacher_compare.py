#!/usr/bin/env python3
"""两个 teacher 的同口径对比:配对指标 + 分区域闪烁 + 锐度。

★ 区域口径见 DATA.md §23:
    发丝 = 轮廓环(dilate12 \ erode12)  —— 不是旧的"边界带",那个混了纯背景
    背景 = ~dilate45
    内部 = erode12
★ 视频是 1024×512 左右拼接(左 GT 右生成),必须对半切(DATA.md §17 的坑)。
★ 锐度 = 对比度归一化梯度,用来排除"更平滑其实是更糊"。
"""
import argparse, glob, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()

import mediapipe as mp
import torch, lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
# ★ net="vgg" 与 inproc_val.py 一致,保证 LPIPS 数字与 val 可比
_lp = lpips_lib.LPIPS(net="vgg").cuda().eval()


def halves(p, n):
    cap = cv2.VideoCapture(p); L, R = [], []
    while len(L) < n:
        ok, f = cap.read()
        if not ok: break
        w = f.shape[1] // 2; L.append(f[:, :w]); R.append(f[:, w:])
    cap.release(); return L, R


def masks(b, stride=2):
    acc = None
    for i, f in enumerate(b):
        if i % stride: continue
        im = cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
        s = seg.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).segmentation_mask > 0.5
        acc = s if acc is None else (acc | s)
    u = acc.astype(np.uint8)
    return {"发丝": (cv2.dilate(u, K(12)) > 0) & ~(cv2.erode(u, K(12)) > 0),
            "背景": ~(cv2.dilate(u, K(45)) > 0),
            "内部": cv2.erode(u, K(12)) > 0}


def flick(b, m):
    g = [cv2.cvtColor(cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.float32) for f in b]
    d = [np.abs(g[i] - g[i - 1]) for i in range(1, len(g))]
    return {k: float(np.median([x[v].mean() for x in d])) for k, v in m.items()}


def sharp(b, mask):
    o = []
    for f in b:
        g = cv2.cvtColor(cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        o.append(float(np.sqrt(gx * gx + gy * gy)[mask].mean() / max(g[mask].std(), 1e-6)))
    return float(np.median(o))


def paired(gt, gen):
    P, S, L = [], [], []
    for a_, b_ in zip(gt, gen):
        A = cv2.cvtColor(a_, cv2.COLOR_BGR2RGB); B = cv2.cvtColor(b_, cv2.COLOR_BGR2RGB)
        mse = float(((A.astype(np.float64) - B.astype(np.float64)) ** 2).mean())
        P.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
        S.append(ssim_fn(A, B, channel_axis=2, data_range=255))
        ta = torch.from_numpy(A).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        tb = torch.from_numpy(B).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
        with torch.no_grad(): L.append(float(_lp(ta, tb)))
    return float(np.mean(P)), float(np.mean(S)), float(np.mean(L))


rows = {}
gt_row = None
for d, nm in zip(a.dirs, a.names):
    fs = sorted(glob.glob(os.path.join(d, "*.mp4")))
    acc, sh, pp, gtacc, gtsh = [], [], [], [], []
    for p in fs:
        gt, gen = halves(p, a.n)
        if len(gt) < 8: continue
        m = masks(gt)
        acc.append(flick(gen, m)); sh.append(sharp(gen, m["发丝"]))
        gtacc.append(flick(gt, m)); gtsh.append(sharp(gt, m["发丝"]))
        pp.append(paired(gt, gen))
    if not acc: print(f"{nm}: 无数据"); continue
    rows[nm] = dict(n=len(acc),
                    发丝=np.median([x["发丝"] for x in acc]),
                    背景=np.median([x["背景"] for x in acc]),
                    内部=np.median([x["内部"] for x in acc]),
                    锐度=np.median(sh),
                    PSNR=np.mean([x[0] for x in pp]),
                    SSIM=np.mean([x[1] for x in pp]),
                    LPIPS=np.mean([x[2] for x in pp]))
    if gt_row is None:
        gt_row = dict(n=len(gtacc), 发丝=np.median([x["发丝"] for x in gtacc]),
                      背景=np.median([x["背景"] for x in gtacc]),
                      内部=np.median([x["内部"] for x in gtacc]),
                      锐度=np.median(gtsh), PSNR=float("nan"), SSIM=float("nan"), LPIPS=float("nan"))

hdr = "%-14s %4s %8s %8s %8s %8s %8s %8s %8s" % (
    "来源", "n", "发丝", "背景", "内部运动", "发丝锐度", "PSNR", "SSIM", "LPIPS")
print(hdr); print("-" * len(hdr))
def show(nm, r):
    print("%-14s %4d %8.3f %8.3f %8.3f %8.3f %8s %8s %8s" % (
        nm, r["n"], r["发丝"], r["背景"], r["内部"], r["锐度"],
        "—" if r["PSNR"] != r["PSNR"] else "%.2f" % r["PSNR"],
        "—" if r["SSIM"] != r["SSIM"] else "%.4f" % r["SSIM"],
        "—" if r["LPIPS"] != r["LPIPS"] else "%.4f" % r["LPIPS"]))
if gt_row: show("GT", gt_row)
for nm, r in rows.items(): show(nm, r)

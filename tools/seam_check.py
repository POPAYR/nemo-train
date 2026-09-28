import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys as _sys
if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
#!/usr/bin/env python3
"""量滑窗接缝跳变:比较接缝帧与其他帧的帧间差。

固定窗口 [0,24)[20,44)[40,64) 的覆盖次数在 t=20,24,40,44 处跳变 ⇒ 这几处最可能出现跳变。
指标:seam_ratio = 接缝处帧差均值 / 非接缝处帧差中位数(>1 = 接缝处更跳)。
同时对 GT(同一段帧)算同样的比值作参照 —— GT 在这几帧没有理由特殊,应≈1。
用法: seam_check.py --dirs A B --names a b [--gt_root ...]
"""
import argparse, glob, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--gt_root", default=XP("XN_HALLO3", "face_frames"))
ap.add_argument("--seams", type=int, nargs="+", default=[20, 24, 40, 44])
ap.add_argument("--n", type=int, default=64)
a = ap.parse_args()


def frames_mp4(p, n):
    cap = cv2.VideoCapture(p); F = []
    while len(F) < n:
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    return F


def diffs(F):
    g = [cv2.cvtColor(cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.float32) for f in F]
    return np.array([np.abs(g[t] - g[t - 1]).mean() for t in range(1, len(g))])   # d[t-1] = |f_t - f_{t-1}|


def ratio(d):
    s = [t - 1 for t in a.seams if 0 < t <= len(d)]
    rest = np.delete(d, s)
    return float(d[s].mean() / np.median(rest))


rows = {}
clips = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.dirs[0], "*.mp4")))
for c in clips:
    key = c.split("_cfg")[0]
    gd = [d for d in glob.glob(os.path.join(a.gt_root, key + "*")) if os.path.isdir(d)]
    if gd:
        fs = sorted(f for f in os.listdir(gd[0]) if f.endswith(".jpg"))[: a.n]
        rows.setdefault("GT", []).append(ratio(diffs([cv2.imread(os.path.join(gd[0], f)) for f in fs])))
    for d, nm in zip(a.dirs, a.names):
        p = os.path.join(d, c)
        if os.path.isfile(p):
            rows.setdefault(nm, []).append(ratio(diffs(frames_mp4(p, a.n))))

print("接缝帧 %s 的帧差 / 其余帧帧差中位数(=1 无跳变)" % a.seams)
print("%-12s %s   %s" % ("来源", "  ".join("clip%d" % i for i in range(len(clips))), "均值"))
for nm, v in rows.items():
    print("%-12s %s   %.3f" % (nm, "  ".join("%5.2f" % x for x in v), np.mean(v)))

#!/usr/bin/env python3
"""眼/嘴放大盲测视频:细微运动肉眼对比用。

输入:若干目录(左 GT 右生成的 mp4,按 clip id 对齐)。对每个 clip:
  - 用 insightface 5 关键点在 **GT 首帧**定位 眼+眉 与 嘴 两个区域(各模型共用同一裁剪框,公平)
    (xnemo 环境的 mediapipe FaceMesh 图解析失败;本脚本用 face 环境运行)
  - 从各模型的生成半幅裁出两区域,放大后竖排(上眼下嘴)
  - 各模型左右顺序按 seed 随机打乱,只标 A/B/C…;答案写到 <out>/_key/key.json(先看视频再看答案)
  - 音频取第一个目录的 mp4;不含 GT(渲染规则 --no_gt)
"""
import argparse, glob, json, os, random, subprocess
import cv2, numpy as np
from insightface.app import FaceAnalysis

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--panel_w", type=int, default=480)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "_key"), exist_ok=True)
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(512, 512))
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"   # 拉丁字母标签;Droid 回退字体没有 A/B 字形


def read(p):
    cap = cv2.VideoCapture(p); L, R = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        w = f.shape[1] // 2; L.append(f[:, :w]); R.append(f[:, w:])
    return L, R


def boxes(k, W, H):
    """k: 5 关键点 [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]。眼框含眉毛,嘴框留出张合余量。"""
    le, re_, _, lm_, rm = k
    d = np.linalg.norm(re_ - le); ec = (le + re_) / 2
    eb = (ec[0] - 0.95 * d, ec[1] - 0.55 * d, ec[0] + 0.95 * d, ec[1] + 0.30 * d)
    mw = np.linalg.norm(rm - lm_); mc = (lm_ + rm) / 2
    mb = (mc[0] - 0.85 * mw, mc[1] - 0.45 * mw, mc[0] + 0.85 * mw, mc[1] + 0.60 * mw)
    cl = lambda b: (int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3])))
    return cl(eb), cl(mb)


def crop(fr, b, w):
    x0, y0, x1, y1 = b; c = fr[y0:y1, x0:x1]
    return cv2.resize(c, (w, int(round(c.shape[0] * w / c.shape[1]))), interpolation=cv2.INTER_CUBIC)


key = {}
fmaps = [{os.path.basename(p)[:12]: p for p in glob.glob(os.path.join(d, "*.mp4"))} for d in a.dirs]
clips = sorted(set.intersection(*[set(m) for m in fmaps]))
for ci, c in enumerate(clips):
    G0, _ = read(fmaps[0][c])
    H, W = G0[0].shape[:2]
    fs = app.get(G0[0])
    if not fs:
        print(f"[skip] {c} 未检测到人脸"); continue
    f0 = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    be, bm = boxes(f0.kps, W, H)
    order = list(range(len(a.dirs))); random.Random(a.seed * 1000 + ci).shuffle(order)
    gens = [read(fmaps[k][c])[1] for k in order]
    n = min(len(g) for g in gens)
    tmp = os.path.join(a.out, f"_{c}.raw.mp4")
    vw = None
    for t in range(n):
        cols = []
        for j, g in enumerate(gens):
            e, m = crop(g[t], be, a.panel_w), crop(g[t], bm, a.panel_w)
            col = np.concatenate([e, np.zeros((6, a.panel_w, 3), np.uint8), m], 0)
            cols.append(col)
        hmax = max(x.shape[0] for x in cols)
        cols = [np.pad(x, ((0, hmax - x.shape[0]), (0, 0), (0, 0))) for x in cols]
        frame = np.concatenate([np.concatenate([x, np.full((hmax, 8, 3), 255, np.uint8)], 1) for x in cols], 1)[:, :-8]
        if vw is None:
            hh, ww = frame.shape[:2]; hh += hh % 2; ww += ww % 2
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (ww, hh))
        frame = np.pad(frame, ((0, hh - frame.shape[0]), (0, ww - frame.shape[1]), (0, 0)))
        vw.write(frame)
    vw.release()
    vf = ",".join(f"drawtext=fontfile={F}:text='{chr(65+j)}':x={j*(a.panel_w+8)+12}:y=10:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.6" for j in range(len(order)))
    outp = os.path.join(a.out, f"{c}_盲测.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", fmaps[0][c], "-vf", vf,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", outp], check=True)
    os.remove(tmp)
    key[c] = {chr(65 + j): a.names[k] for j, k in enumerate(order)}
    print(f"[ok] {c} -> {os.path.basename(outp)}", flush=True)
json.dump(key, open(os.path.join(a.out, "_key", "key.json"), "w"), ensure_ascii=False, indent=1)
print(f"[done] {len(key)} 条 -> {a.out}   答案在 _key/key.json")

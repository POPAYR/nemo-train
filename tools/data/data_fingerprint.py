#!/usr/bin/env python3
"""数据指纹:确认实验机从原始 mp4 重新处理出的数据与开发机一致。

对固定一批 clip 记录:帧数、若干帧像素均值/标准差、frame_latent / pose_embed_real 的形状与统计量、音频长度。
  生成参考(开发机):python tools/data/data_fingerprint.py --out docs/data_fingerprint_ref.json
  比对(实验机):    python tools/data/data_fingerprint.py --ref docs/data_fingerprint_ref.json
容差说明:帧是 JPEG,检测/缩放在不同 GPU/库版本下可能有亚像素差,故用统计量 + 宽松阈值;
帧数、形状、音频长度必须**完全**相等(它们决定对齐)。
"""
import argparse, json, os, random, sys
import numpy as np, torch
from PIL import Image
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP

ap = argparse.ArgumentParser()
ap.add_argument("--out"); ap.add_argument("--ref"); ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
D = XP("XN_HALLO3")


def fp(nm):
    fd = f"{D}/face_frames/{nm}"
    fs = sorted(f for f in os.listdir(fd) if f.endswith(".jpg"))
    pick = [0, len(fs) // 2, len(fs) - 1]
    px = [np.asarray(Image.open(f"{fd}/{fs[i]}").convert("RGB"), np.float32) for i in pick]
    lat = torch.load(f"{D}/frame_latent/{nm}.pt", map_location="cpu").float()
    pe = torch.load(f"{D}/pose_embed_real/{nm}.pt", map_location="cpu").float()
    au = torch.load(f"{D}/audio_pt/{nm}.pt", map_location="cpu")
    return dict(nf=len(fs), px_mean=[float(x.mean()) for x in px], px_std=[float(x.std()) for x in px],
                lat_shape=list(lat.shape), lat_mean=float(lat.mean()), lat_std=float(lat.std()),
                pe_shape=list(pe.shape), pe_mean=float(pe.mean()), pe_std=float(pe.std()), audio_n=int(au.numel()))


if a.out:
    tr = [l.strip() for l in open(f"{D}/train_data_ge64.txt") if l.strip()]
    ts = [l.strip() for l in open(f"{D}/testset_clips.txt") if l.strip()]
    random.seed(20260928)
    clips = sorted(random.sample(tr, a.n)) + sorted(ts)[:10]
    json.dump({c: fp(c) for c in clips}, open(a.out, "w"), indent=1)
    print(f"[fingerprint] 写出 {len(clips)} 条参考 → {a.out}")
else:
    ref = json.load(open(a.ref)); bad = 0; rows = []
    for c, r in ref.items():
        try: x = fp(c)
        except Exception as e:
            print(f"  ✗ {c}: 读取失败 {type(e).__name__} {e}"); bad += 1; continue
        hard = [k for k in ("nf", "lat_shape", "pe_shape", "audio_n") if x[k] != r[k]]
        dpx = max(abs(p - q) for p, q in zip(x["px_mean"], r["px_mean"]))
        dlat = abs(x["lat_mean"] - r["lat_mean"]) / (abs(r["lat_std"]) + 1e-6)
        dpe = abs(x["pe_mean"] - r["pe_mean"]) / (abs(r["pe_std"]) + 1e-6)
        ok = not hard and dpx < 2.0 and dlat < 0.02 and dpe < 0.05
        bad += (not ok); rows.append((c, ok, hard, dpx, dlat, dpe))
    for c, ok, hard, dpx, dlat, dpe in rows:
        print(f"  {'✓' if ok else '✗'} {c[:12]}  硬不一致={hard or '-'}  像素均值差={dpx:.2f}  latent偏移={dlat:.4f}σ  pose偏移={dpe:.4f}σ")
    print(f"[fingerprint] {len(ref) - bad}/{len(ref)} 一致" + ("" if not bad else "  ✗ 有不一致,先别训练,把这段输出发给 Claude"))
    sys.exit(1 if bad else 0)

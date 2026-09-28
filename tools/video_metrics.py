import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys as _sys
if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
#!/usr/bin/env python3
"""多个 teacher 的配对/分布指标:PSNR SSIM LPIPS FID FVD,同一批 clip。

输入:若干目录,每个目录里是 1024×512 左 GT 右生成的 mp4,按文件名前 12 字符(clip id)对齐,
      只统计**所有目录都有**的 clip。
口径(与 src/utils/inproc_val.py 一致):
  PSNR/SSIM:512×512 RGB 逐帧;LPIPS:net=vgg,[-1,1]
  FID:pytorch_fid,2048 维,生成帧 vs GT 帧(全部帧)
  FVD16:styleganv i3d,每 clip 前 16 帧 ——与训练内 val 同口径
  FVD64:每 clip 切成 4 段不重叠的 16 帧(覆盖全部 64 帧,含滑窗接缝)
"""
import argparse, glob, os, shutil, sys
import cv2, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--work", required=True, help="FID 临时 png 目录(放 repo 内)")
a = ap.parse_args()
dev = "cuda"

import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
from pytorch_fid import fid_score
sys.path.insert(0, third_party("fvd"))
from calculate_fvd import calculate_fvd
_lp = lpips_lib.LPIPS(net="vgg").to(dev).eval()


def halves(p, n):
    cap = cv2.VideoCapture(p); L, R = [], []
    while len(L) < n:
        ok, f = cap.read()
        if not ok: break
        w = f.shape[1] // 2
        L.append(cv2.cvtColor(f[:, :w], cv2.COLOR_BGR2RGB)); R.append(cv2.cvtColor(f[:, w:], cv2.COLOR_BGR2RGB))
    return np.stack(L), np.stack(R)


def fvd(A, B, T=16, size=224):
    def t(V):
        x = torch.from_numpy(np.stack(V)).float().permute(0, 1, 4, 2, 3) / 255
        return torch.nn.functional.interpolate(x.flatten(0, 1), (size, size), mode="bilinear",
                                               align_corners=False).view(len(V), T, 3, size, size)
    r = calculate_fvd(t(A), t(B), device=dev, method="styleganv")["value"]
    return float(list(r.values())[-1] if isinstance(r, dict) else r[-1])


key = lambda p: os.path.basename(p)[:12]
files = [{key(p): p for p in glob.glob(os.path.join(d, "*.mp4"))} for d in a.dirs]
clips = sorted(set.intersection(*[set(f) for f in files]))
print(f"[clips] 共同 clip {len(clips)} 条", flush=True)

rows = {}
gt_dir = os.path.join(a.work, "_gt")
for di, (fm, nm) in enumerate(zip(files, a.names)):
    gdir = os.path.join(a.work, nm)
    shutil.rmtree(gdir, ignore_errors=True); os.makedirs(gdir)
    if di == 0:
        shutil.rmtree(gt_dir, ignore_errors=True); os.makedirs(gt_dir)
    P, S, L, V16g, V16r, V64g, V64r = [], [], [], [], [], [], []
    for c in clips:
        G, R = halves(fm[c], a.n)
        n = min(len(G), len(R))
        for j in range(n):
            mse = float(((R[j].astype(np.float64) - G[j].astype(np.float64)) ** 2).mean())
            P.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
            S.append(ssim_fn(G[j], R[j], channel_axis=2, data_range=255))
            with torch.no_grad():
                x = torch.from_numpy(R[j]).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                y = torch.from_numpy(G[j]).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                L.append(float(_lp(x, y)))
            cv2.imwrite(os.path.join(gdir, f"{c}_{j:03d}.png"), cv2.cvtColor(R[j], cv2.COLOR_RGB2BGR))
            if di == 0:
                cv2.imwrite(os.path.join(gt_dir, f"{c}_{j:03d}.png"), cv2.cvtColor(G[j], cv2.COLOR_RGB2BGR))
        V16g.append(R[:16]); V16r.append(G[:16])
        for s in range(0, n - 15, 16):
            V64g.append(R[s:s + 16]); V64r.append(G[s:s + 16])
    fid = fid_score.calculate_fid_given_paths([gdir, gt_dir], batch_size=32, device=dev, dims=2048)
    rows[nm] = dict(PSNR=np.mean(P), SSIM=np.mean(S), LPIPS=np.mean(L), FID=fid,
                    FVD16=fvd(V16g, V16r), FVD64=fvd(V64g, V64r), nfr=len(P))
    print(f"[done] {nm}: " + " ".join(f"{k}={v:.4f}" for k, v in rows[nm].items()), flush=True)

print("\n%-22s %7s %7s %7s %7s %7s %7s" % ("来源", "PSNR", "SSIM", "LPIPS", "FID", "FVD16", "FVD64"))
for nm, r in rows.items():
    print("%-22s %7.2f %7.4f %7.4f %7.2f %7.1f %7.1f" % (nm, r["PSNR"], r["SSIM"], r["LPIPS"], r["FID"], r["FVD16"], r["FVD64"]))
print(f"(每行 {len(clips)} clip × 64 帧;FVD16=每 clip 前16帧,与训练内 val 同口径;FVD64=4 段×16 帧,覆盖接缝)")

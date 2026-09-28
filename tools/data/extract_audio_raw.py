#!/usr/bin/env python3
"""从原始 mp4 抽完整音轨 → audio_wav_raw/<clip>.wav(16kHz 单声道 pcm_s16le)。

这是 recut_audio.py 的输入。开发机上 audio_wav_raw 与此法重抽逐采样点一致(最大差 0.0,2026-09-28 实测)。
用法: python tools/data/extract_audio_raw.py <clip清单> [--workers 32]
"""
import argparse, os, subprocess, sys
from multiprocessing import Pool
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP

ap = argparse.ArgumentParser()
ap.add_argument("list"); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
RAW, OUT = XP("XN_HALLO3_RAW"), XP("XN_HALLO3", "audio_wav_raw")
os.makedirs(OUT, exist_ok=True)
names = [l.strip() for l in open(a.list) if l.strip()]


def one(nm):
    dst = f"{OUT}/{nm}.wav"
    if os.path.exists(dst) and os.path.getsize(dst) > 44: return None
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{RAW}/{nm}.mp4", "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", dst], capture_output=True)
    return None if r.returncode == 0 else f"{nm}: {r.stderr.decode()[:120]}"


with Pool(a.workers) as p:
    errs = [e for e in p.imap_unordered(one, names, chunksize=16) if e]
print(f"[audio_raw] {len(names) - len(errs)}/{len(names)} 条完成,失败 {len(errs)}")
for e in errs[:10]: print("  ", e)
sys.exit(1 if len(errs) > len(names) * 0.01 else 0)

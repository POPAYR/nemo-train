"""按重采样后的新帧数重切音频。

★ 为什么必须重切:帧提取改为按时间轴重采样到 25fps 后,帧数变了
  (24fps 素材 49→51 帧、50fps 194→97 帧)。而 audio_pt/audio_wav 是按**旧帧数**切的,
  不重切就会重新出现音画长度不一致 —— 正是这次要修的问题本身。
  audio_wav_raw 是完整原始音频(时长==视频时长),从它截前 nf/25 秒即可,无需重解码视频。
"""
import os, sys, json, numpy as np, torch, soundfile as sf
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
from multiprocessing import Pool
D = XP("XN_HALLO3")
SR = 16000
names = [l.strip() for l in open(sys.argv[1] if len(sys.argv) > 1 else f"{D}/train_data.txt") if l.strip()]

def one(nm):
    fd = f"{D}/face_frames/{nm}"; raw = f"{D}/audio_wav_raw/{nm}.wav"
    if not (os.path.isdir(fd) and os.path.exists(raw)): return (nm, "缺帧或缺原始音频")
    try:
        nf = len([f for f in os.listdir(fd) if f.endswith(".jpg")])
        w, sr = sf.read(raw, dtype="float32", always_2d=False)
        if sr != SR: return (nm, f"采样率{sr}")
        if w.ndim > 1: w = w.mean(1)
        need = nf * (SR // 25)                       # 每帧 640 采样点
        if len(w) < need:                            # 原始音频略短(容器时长舍入),末尾补零
            if len(w) < need * 0.95: return (nm, "音频过短")
            w = np.pad(w, (0, need - len(w)))
        w = w[:need]
        torch.save(torch.from_numpy(w.copy()), f"{D}/audio_pt/{nm}.pt")
        sf.write(f"{D}/audio_wav/{nm}.wav", w, SR)
        return (nm, None)
    except Exception as e:
        return (nm, f"{type(e).__name__}")

if __name__ == "__main__":
    with Pool(48) as pool:
        res = pool.map(one, names, chunksize=100)
    bad = {n: r for n, r in res if r}
    json.dump(bad, open(f"{D}/recut_audio_failed.json", "w"))
    print(f"[done] 重切 {len(res)-len(bad)}/{len(res)} 条,失败 {len(bad)}", flush=True)
    from collections import Counter
    for k, v in Counter(bad.values()).most_common(): print(f"   {k}: {v}")

"""滑窗(context)调度。

official: 逐行复刻 X-NeMo 官方 src/pipelines/context.py::uniform +
  pipeline_pose2vid_motenc_long.py 的调用方式(step=0, context_stride=1, closed_loop=True):
  ① **每个去噪步**重新抽 offset ~ U[0, W-1]  ② 窗口对 F 取模,首尾环接
  ⇒ 接缝位置每步都变,某一步的接缝在下一步落进窗口内部被抹平。
fixed: 我们 2026-09 之前的实现:窗口位置 20 步不变 ⇒ 接缝固定在同几帧,误差逐步累积成可见跳变。
"""
import random


def official_windows(F, W, overlap, rng: random.Random):
    if F <= W:
        return [list(range(F))]
    off = rng.randint(0, W - 1)               # 官方: random.randint(0, context_frames-1)
    return [[(e + off) % F for e in range(j, j + W)]
            for j in range(0, F, W - overlap)]


def fixed_windows(F, W, overlap):
    if F <= W:
        return [list(range(F))]
    st = max(1, W - overlap)
    wins = [(o, min(o + W, F)) for o in range(0, max(F - W, 0) + 1, st)]
    if wins[-1][1] < F:
        wins.append((F - W, F))
    return [list(range(s0, s1)) for s0, s1 in wins]


def windows(mode, F, W, overlap, rng):
    if not W or W >= F:
        return [list(range(F))]
    return official_windows(F, W, overlap, rng) if mode == "official" else fixed_windows(F, W, overlap)

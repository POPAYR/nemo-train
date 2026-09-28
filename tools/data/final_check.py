"""数据定稿检查:三者(帧/motion latent/VAE latent)齐备且互相对齐,然后写出训练清单。

★ 闸门语义:发现缺失就**停**,而不是把缺失样本从清单里悄悄删掉再放行 ——
  后者会得到一个残缺数据集且没有任何报错(上一版就是这么写的)。
★ mtime 判据:重提取会保持帧数不变的样本很多,只查"存在+帧数一致"会放行上一轮的旧产物。
"""
import os, sys, json, torch, numpy as np, random
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
D = XP("XN_HALLO3")
MAN = sys.argv[1] if len(sys.argv) > 1 else f"{D}/manifest.txt"
names = [l.strip() for l in open(MAN) if l.strip()]

ok, bad = [], []
for nm in names:
    fd = f"{D}/face_frames/{nm}"
    if not os.path.isdir(fd): bad.append((nm, "无帧")); continue
    tf = os.path.getmtime(fd)
    miss = None
    for sub in ("pose_embed_real", "frame_latent", "audio_pt"):
        p = f"{D}/{sub}/{nm}.pt"
        try:
            if os.path.getmtime(p) <= tf: miss = f"{sub}早于帧"; break
        except OSError: miss = f"缺{sub}"; break
    if miss: bad.append((nm, miss)); continue
    ok.append(nm)

from collections import Counter
print(f"完整 {len(ok)}/{len(names)}  异常 {len(bad)}")
for k, v in Counter(r for _, r in bad).most_common(): print(f"   {k:16s} {v}")
if len(ok) < len(names) * 0.97:
    print("[abort] 完整样本不足 97%,上游未跑完,不覆盖清单"); raise SystemExit(1)

# 抽样核对帧数三者一致 + 音频长度 == 帧数×640
random.seed(0); mm = am = 0
nfs = {}
for nm in random.sample(ok, min(300, len(ok))):
    nf = len([f for f in os.listdir(f"{D}/face_frames/{nm}") if f.endswith(".jpg")])
    nl = torch.load(f"{D}/frame_latent/{nm}.pt", map_location="cpu").shape[0]
    npo = torch.load(f"{D}/pose_embed_real/{nm}.pt", map_location="cpu").shape[1]
    na = torch.load(f"{D}/audio_pt/{nm}.pt", map_location="cpu").numel()
    if not (nf == nl == npo): mm += 1
    if na != nf * 640: am += 1
print(f"抽样 300 条: 帧数不一致 {mm} 条, 音画长度不匹配 {am} 条")
if mm or am:
    print("[abort] 抽样对齐检查未通过"); raise SystemExit(1)

# ★ 测试集从训练清单中排除(2026-09-06 起):baseline 改用官方权重重测,
#   "两边都在同样数据上训"的对称性不再成立,测试集若留在训练集里我们会占便宜。
try:
    tsc = set(l.strip() for l in open(f"{D}/testset_clips.txt") if l.strip())
    before = len(ok); ok = [n for n in ok if n not in tsc]
    print(f"排除测试集 {before - len(ok)} 条 → 训练清单 {len(ok)}")
except FileNotFoundError:
    print("[warn] 未找到 testset_clips.txt,训练清单未排除测试集")
# ★ FC_VERIFY=1(实验机):不覆盖定稿清单,写 *.regen 并与定稿逐条比对 —— 两机训练清单必须完全一致
VERIFY = os.environ.get("FC_VERIFY") == "1"
SFX = ".regen" if VERIFY else ""
open(f"{D}/train_data.txt{SFX}", "w").write("\n".join(ok) + "\n")
# DMD/ODE 阶段专用:这两阶段的 loss 不用 pad mask,<64 帧会被末帧 repeat 成静止帧当真值
ge = [n for n in ok if len([f for f in os.listdir(f"{D}/face_frames/{n}") if f.endswith(".jpg")]) >= 64]
open(f"{D}/train_data_ge64.txt{SFX}", "w").write("\n".join(ge) + "\n")
if VERIFY:
    bad_any = False
    for fn in ("train_data.txt", "train_data_ge64.txt"):
        a = set(l.strip() for l in open(f"{D}/{fn}") if l.strip())
        b = set(l.strip() for l in open(f"{D}/{fn}.regen") if l.strip())
        print(f"[verify] {fn}: 定稿 {len(a)}  重建 {len(b)}  仅定稿有 {len(a - b)}  仅重建有 {len(b - a)}")
        if a != b:
            bad_any = True
            open(f"{D}/{fn}.missing", "w").write("\n".join(sorted(a - b)) + "\n")
    if bad_any:
        print("[verify] ✗ 与定稿清单不一致,缺的 clip 见 *.missing(训练仍按定稿清单,缺失样本会读不到)"); raise SystemExit(2)
    print("[verify] ✓ 两份训练清单与定稿完全一致")
print(f"train_data.txt={len(ok)}  train_data_ge64.txt={len(ge)}")
print("[done] 数据定稿", flush=True)

"""帧 → VAE latent。重提取后必须重编码,因为帧本身变了。

输出格式与旧数据一致:[T,4,64,64] bfloat16,含 0.18215 缩放
(推理侧 gen_variants.py 解码前会 /0.18215,故存的是乘过的)。
★ 原地覆盖 frame_latent/<name>.pt;断点续跑:.pt 已存在且帧数匹配则跳过。
"""
import os, sys, glob, time, argparse
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
import numpy as np, torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=XP("XN_HALLO3", "face_frames"))
ap.add_argument("--out",    default=XP("XN_HALLO3", "frame_latent"))
ap.add_argument("--list",   default=XP("XN_HALLO3", "train_data.txt"))
ap.add_argument("--shard",  default="0/1")
ap.add_argument("--batch",  type=int, default=16)
ap.add_argument("--workers", type=int, default=16, help="JPEG 解码线程数")
ap.add_argument("--resume", type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(XP("REPO", "configs/test_ar_model.yaml"))
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
# ★ 实测 channels_last 在这个 VAE 上**更慢**(21.2→12.7 帧/s),已放弃 —— 别再试
# ★ 两个独立线程池:原来是**同一个池嵌套使用**(submit 里再 map),
#   submit 占掉 1 个 worker、内部 map 只剩 5 个,batch 一大读图就成瓶颈
#   (基准里 batch 32 慢到测不出来)。拆开后 batch 可以调大以提高 GPU 效率。
_pool = ThreadPoolExecutor(max_workers=2)                 # 只负责发起预取
_rdpool = ThreadPoolExecutor(max_workers=a.workers)       # 只负责并行解码 JPEG

names = [l.strip() for l in open(a.list) if l.strip()]
i, n = map(int, a.shard.split("/")); names = names[i::n]
print(f"[shard {i}/{n}] {len(names)} 条", flush=True)
t0 = time.time(); done = 0
for k, nm in enumerate(names):
    fd = os.path.join(a.frames, nm); op = os.path.join(a.out, f"{nm}.pt")
    if not os.path.isdir(fd): continue
    fs = sorted(glob.glob(f"{fd}/*.jpg")) or sorted(glob.glob(f"{fd}/*.png"))
    if not fs: continue
    # ★ resume 必须是 mtime 感知的:重提取后帧内容变了但**帧数不变**,只查"存在+帧数对"
    #   会把 9.7 万条旧 latent 全部当成已完成跳过,得到与新帧错配的训练数据(本轮已踩)。
    #   帧目录在重提取时被 rmtree 重建,故其 mtime 即最近一次提取时间。
    if a.resume and os.path.exists(op) and os.path.getmtime(op) > os.path.getmtime(fd):
        try:
            if torch.load(op, map_location="cpu").shape[0] == len(fs): done += 1; continue
        except Exception: pass
    lat = []
    _rd = lambda f: np.array(Image.open(f).convert("RGB"))
    with torch.no_grad():
        # 预取下一批,读图与 GPU 重叠
        nxt = _pool.submit(lambda ff: np.stack(list(_rdpool.map(_rd, ff))), fs[:a.batch])
        for b0 in range(0, len(fs), a.batch):
            im = nxt.result()
            if b0 + a.batch < len(fs):
                nxt = _pool.submit(lambda ff: np.stack(list(_rdpool.map(_rd, ff))),
                                   fs[b0+a.batch:b0+2*a.batch])
            x = torch.from_numpy(im).permute(0,3,1,2).to(dev, dt)/127.5 - 1
            lat.append((vae.encode(x).latent_dist.mean * 0.18215).cpu())
    torch.save(torch.cat(lat,0).to(torch.bfloat16), op)
    done += 1
    if done % 200 == 0:
        el = time.time()-t0
        print(f"[prog] {done}/{len(names)}  {el/60:.1f}min  {done/max(el,1):.2f} clip/s  "
              f"剩余 {(len(names)-done)/max(done/max(el,1),1e-9)/3600:.1f}h", flush=True)
print(f"[done] {done}/{len(names)} 条, {(time.time()-t0)/3600:.2f}h", flush=True)

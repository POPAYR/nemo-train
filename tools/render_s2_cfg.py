"""用指定 CFG 渲染 stage2 视频(验证器本身没有 CFG 分支,固定 cfg=1.0)。

uncond 构造与训练侧 cfg_drop 逐条一致:
  ① reference bank 不注入  ② CLIP 置零  ③ motion 换成参考帧的
外推式与官方 pipeline 一致: v = v_u + w*(v_c - v_u)
"""
import os, sys, argparse, time
import numpy as np, torch
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from src.models.mutual_self_attention import ReferenceAttentionControl

ap = argparse.ArgumentParser()
ap.add_argument("--s2_ckpt", required=True)
ap.add_argument("--stage1_ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cfgs", type=float, nargs="+", default=[2.0])  # ★ 2026-09-21 定稿交付档
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--window", type=int, default=0, help="0=单窗整段;>0=滑窗长度")
ap.add_argument("--overlap", type=int, default=4)
ap.add_argument("--ctx", choices=["official", "fixed"], default="official",
                help="滑窗调度:official=官方逐步随机偏移+环形(默认);fixed=旧的固定窗口(有接缝跳变)")
ap.add_argument("--no_gt", action="store_true", help="只输出生成画面(默认左 GT 右生成,供 band_flicker 用)")
ap.add_argument("--audio_dir", default=XP("XN_HALLO3", "audio_wav"),
                help="<clip>.wav 与帧 25fps 对齐;空串=不合音频")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--data_seed", type=int, default=1234,
                help="Validator 的噪声 seed(每 clip = data_seed + 序号);拆分 clip 清单并行时用它对齐序号偏移")
ap.add_argument("--use_ema", action="store_true", help="用 ckpt 的 ema 键而非瞬时权重")
ap.add_argument("--clips", type=int, default=4)
ap.add_argument("--rope", action="store_true", default=True)
a = ap.parse_args()
dev = torch.device("cuda"); dt = torch.bfloat16
sys.argv = [sys.argv[0]]
from scripts.train.flow_stage2_temporal import build
from src.utils.inproc_val import Validator

refu, denu, imgenc, _ = build(dev, dt, a.stage1_ckpt, a.rope)
_ck = torch.load(a.s2_ckpt, map_location="cpu")
# ★ --use_ema:优先用 EMA 权重(我们的最佳权重都在 ema 键里;瞬时权重在小 batch 下会游走)
if a.use_ema and isinstance(_ck, dict) and "ema" in _ck:
    sd = dict(_ck["denoising_unet"]); sd.update(_ck["ema"])
    print("[build] 使用 EMA 权重(%d 个张量覆盖)" % len(_ck["ema"]), flush=True)
else:
    sd = _ck.get("denoising_unet", _ck)
    if a.use_ema:
        print("[build] ⚠ ckpt 里没有 ema 键,退回瞬时权重", flush=True)
denu.load_state_dict(sd, strict=False); denu.eval()
print(f"[build] s2={os.path.basename(a.s2_ckpt)}  s1={os.path.basename(a.stage1_ckpt)}", flush=True)
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read",  batch_size=1, fusion_blocks="full")
cfgc = OmegaConf.load(XP("REPO", "configs/test_ar_model.yaml"))
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfgc.vae_path).to(dev, dt).eval()

V = Validator(out_dir=os.path.join(a.out, "_val"), kind="video", seed=a.data_seed, n_clips=a.clips, n_frames=0,
              sample_steps=a.steps, sample_shift=a.shift, is_main=True)
items = V._load(dev, dt)
t = torch.linspace(1.0, 0.0, a.steps + 1, device=dev)
sig = a.shift * t / (1 + (a.shift - 1) * t) if a.shift > 1 else t

def set_ref(it, ce):
    rw.clear()
    refu(it["ref_lat"], torch.zeros((), device=dev).long(), encoder_hidden_states=ce, return_dict=False)
    rr.update(rw, dtype=dt)

@torch.no_grad()
def sample(it, ce, w):
    z = it["eps"].to(dt).clone(); mo_c = it["mo"]
    mo_u = mo_c[:, 0:1].expand_as(mo_c).contiguous(); ce_u = torch.zeros_like(ce)
    # ★ 滑窗调度见 src/utils/ctx_sched.py(与官方一致:每步随机偏移 + 环形);--ctx fixed 复现旧行为。
    F_ = z.shape[2]
    import random as _random
    from src.utils.ctx_sched import windows as _windows
    _rng = _random.Random(a.seed)
    for i in range(a.steps):
        tt = torch.full((1,), float(sig[i]) * 1000.0, device=dev, dtype=dt)
        v = torch.zeros_like(z, dtype=torch.float32)
        cnt = torch.zeros((1, 1, F_, 1, 1), device=dev, dtype=torch.float32)
        for _ix in _windows(a.ctx, F_, a.window, a.overlap, _rng):
            s0 = torch.as_tensor(_ix, device=z.device)
            zw = z[:, :, s0]; mw_c = mo_c[:, s0]; mw_u = mo_u[:, s0]
            set_ref(it, ce)
            vw = denu(zw, tt, encoder_hidden_states=[ce, mw_c], pose_cond_fea=None, return_dict=False)[0].float()
            if w != 1.0:
                rr.clear()
                vw_u = denu(zw, tt, encoder_hidden_states=[ce_u, mw_u], pose_cond_fea=None, return_dict=False)[0].float()
                vw = vw_u + w * (vw - vw_u)
            v[:, :, s0] += vw
            cnt[:, :, s0] += 1.0
        v = v / cnt.clamp_min(1.0)
        z = (z.float() + (float(sig[i+1]) - float(sig[i])) * v).to(dt)
    x = rearrange(z, "b c f h w -> (b f) c h w") / 0.18215
    dec = torch.cat([vae.decode(x[i:i+8], x[i:i+8].shape[0]).sample for i in range(0, x.shape[0], 8)], 0)
    return ((dec/2+0.5).clamp(0,1).permute(0,2,3,1).cpu().float().numpy()*255).astype(np.uint8)

from PIL import Image
import cv2
os.makedirs(a.out, exist_ok=True)
for ci, it in enumerate(items):
    ce = V._clip_emb(imgenc, it["ref_path"], dev, dt)
    gfs = sorted(os.listdir(it["gt_dir"]))
    GT = np.stack([np.array(Image.open(os.path.join(it["gt_dir"], gfs[j])).convert("RGB").resize((512,512)))
                   for j in it["idx"]])
    for w in a.cfgs:
        t0 = time.time(); img = sample(it, ce, w)
        n = min(len(GT), len(img))
        arr = img[:n] if a.no_gt else np.concatenate([GT[:n], img[:n]], 2)   # 默认左 GT 右生成
        p = f"{a.out}/{it['clip'][:12]}_cfg{w}.mp4"
        _wav = os.path.join(a.audio_dir, it["clip"] + ".wav") if a.audio_dir else ""
        _t0 = float(it["idx"][0]) / 25.0     # 音频从片段首帧对应的时间点截取
        # ★ 必须与 tool/render_eps_teacher.py 用同一编码器(libx264 crf16)。
        #   曾用 cv2 的 mp4v,压缩噪声显著虚增帧间差,导致与 ε teacher 的对比不可比。
        import subprocess
        pr = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                               "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r","25","-i","-"] +
                              (["-ss", f"{_t0:.3f}", "-i", _wav, "-map", "0:v:0", "-map", "1:a:0",
                                "-c:a", "aac", "-b:a", "128k", "-shortest"] if _wav and os.path.isfile(_wav) else []) +
                              ["-c:v","libx264","-pix_fmt","yuv420p","-crf","16", p], stdin=subprocess.PIPE)
        pr.stdin.write(arr.tobytes()); pr.stdin.close(); pr.wait()
        print(f"  {it['clip'][:10]} cfg={w} {n}帧 {time.time()-t0:.0f}s -> {os.path.basename(p)}", flush=True)
print(f"[done] -> {a.out}", flush=True)

"""
验证 ref-only score(s_real)是否学到"真实脸是动态的"先验（方案② 成立与否）。
=====================================================================================
DMD 视角：s_real 惩罚静态 x̂ ⟺ 对「静态输入」加噪后，s_real 预测的 x0 会"脑补动态"
（temporal std > 0）→ DMD 梯度 (x0_fake−x0_real) 把静态往动态推。
测：给 s_real 喂 [真实动态视频] 和 [静态视频(首帧重复)] 加噪，比预测 x0 的帧间方差(temporal std)。
判据：static 输入的 x0 帧间方差**明显 > 0 且接近 dynamic** → s_real 有动态先验 → 方案②成立。
用法：CUDA_VISIBLE_DEVICES=0 python scripts/val/verify_refonly.py --ckpt .../refonly_step_6000.pt
"""
import os, sys, argparse
import numpy as np
import torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.distill.models import DMD2Models
from src.distill.dmd_loss import eps_to_x0
from data.dataset import MotarDataset

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
VAE_SCALE = 0.18215


def tstd(x):  # x:[B,C,F,H,W] → 帧间标准差(over F)，再平均，衡量"动态程度"
    return x.float().std(dim=2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device("cuda:0"); dt = torch.bfloat16

    M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=8)
    sd = torch.load(args.ckpt, map_location="cpu")["critic"]
    M.critic.load_state_dict(sd, strict=True); M.critic.eval().requires_grad_(False)
    from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
    cfg = OmegaConf.load(TRAIN_CFG)
    mot_enc = MotEncoder().to(dev, dt).eval().requires_grad_(False)
    mot_enc.load_state_dict(torch.load(cfg.motion_encoder_path, map_location="cpu"), strict=True)
    print(f"[verify] ckpt={os.path.basename(args.ckpt)}  n={args.n_samples}")

    s = cfg.data.sources[0]
    from diffusers.video_processor import VideoProcessor
    ds = MotarDataset(pose_dir=s.pose_dir, audio_dir=s.audio_dir, caption_dir=s.caption_dir,
                      data_name_path=s.data_name_path, tokenizer_path=cfg.data.tokenizer_path,
                      data_stats_path=cfg.data.data_stats_path, context_length=args.frames,
                      fps=cfg.data.fps, sr=cfg.data.sr, text_max_len=128, random_crop=True,
                      pad_short=True, load_video=True, latent_dir=s.latent_dir, video_dir=s.video_dir,
                      video_processor=VideoProcessor(do_resize=True, vae_scale_factor=8))
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)
    it = iter(dl)

    T_LIST = [200, 500, 800]
    agg = {t: {"dyn_in": [], "dyn_x0": [], "sta_in": [], "sta_x0": []} for t in T_LIST}
    with torch.no_grad():
        for _ in range(args.n_samples):
            b = next(it)
            x_dyn = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous() * VAE_SCALE  # [1,C,F,H,W]
            x_static = x_dyn[:, :, 0:1].repeat(1, 1, x_dyn.shape[2], 1, 1)                          # 首帧重复=静态
            ref_lat = b["ref_latent"].to(dev, dt) * VAE_SCALE
            clip = M.clip_embed(b["ref_img"].to(dev, dt))
            B, C, F_, H, W = x_dyn.shape
            M.set_reference(ref_lat, clip, B)
            rmc = b["ref_mot_cond"].to(dev, dt)
            bbox = torch.ones((B, 3), device=dev, dtype=dt); bbox[:, :2] = 0
            neg = mot_enc(rmc, bbox)
            null_mot = neg.unsqueeze(1).expand(B, F_, *neg.shape[1:]).to(dt)
            for t_i in T_LIST:
                t = torch.full((B,), t_i, device=dev, dtype=torch.long)
                noise = torch.randn_like(x_dyn)
                for tag, x in [("dyn", x_dyn), ("sta", x_static)]:
                    x_t = M.scheduler.add_noise(x, noise, t).to(dt)
                    eps, _ = M.forward_net(M.critic, x_t, t, clip, null_mot)
                    x0 = eps_to_x0(x_t, eps, t, M.acp)
                    agg[t_i][f"{tag}_in"].append(tstd(x))
                    agg[t_i][f"{tag}_x0"].append(tstd(x0))

    print(f"\n{'t':>5} | {'dyn_in':>8} {'dyn_x0':>8} | {'sta_in':>8} {'sta_x0':>8} | 判读")
    print("-" * 66)
    for t_i in T_LIST:
        a = agg[t_i]
        di, dx, si, sx = [np.mean(a[k]) for k in ("dyn_in", "dyn_x0", "sta_in", "sta_x0")]
        verdict = "✅静态被脑补出动态" if sx > 3 * max(si, 1e-4) else "⚠️静态x0仍偏静"
        print(f"{t_i:>5} | {di:8.4f} {dx:8.4f} | {si:8.4f} {sx:8.4f} | {verdict}")
    print("\n判据：sta_x0(s_real 对静态的 x0 帧间方差) 明显 > sta_in(≈0) 且接近 dyn → s_real 有动态先验、方案②成立")


if __name__ == "__main__":
    main()

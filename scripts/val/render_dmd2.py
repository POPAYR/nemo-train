"""
渲染 DMD2 student：ode-baseline(4步) vs dmd2(4步) vs dmd2(1步)，看分布匹配是否让少步更锐 / 1步可用。
用 self-forcing rollout（因果少步、从噪声生成 = student 实际部署态），SVD VAE 解码（高质量）。
用法: CUDA_VISIBLE_DEVICES=N python scripts/val/render_dmd2.py --dmd2_ckpt output/dmd2/dmd2_step_2000.pt
"""
import sys, os, argparse, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models, DEC_CFG
from src.distill.rollout import self_forcing_rollout
from src.utils.util import save_videos_grid

ap = argparse.ArgumentParser()
ap.add_argument("--dmd2_ckpt", default=None)
ap.add_argument("--frames", type=int, default=32)
ap.add_argument("--sample", default="M003_video_down_angry_level_1_001")
ap.add_argument("--out", default="/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_render")
args = ap.parse_args()
dev = torch.device("cuda:0")

M = DMD2Models(dev, gen_ckpt="/media/ps/ssd5/ayr/x-nemo-inference/output/ode_init/ode_step_8000.pt", block_size=8)
cfg = OmegaConf.load(DEC_CFG)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, M.dt).eval()
clip_proc = CLIPImageProcessor()

ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"; n = args.sample; F_ = args.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip_pix = clip_proc.preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, M.dt)
clip = M.image_encoder(clip_pix).image_embeds.unsqueeze(1)
ref_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()[0:1].to(dev, M.dt)
motion = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float()[:, :F_].reshape(1, F_, 32, 16).to(dev, M.dt)
g = torch.Generator(device=dev); g.manual_seed(1234)
noise = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=M.dt)
os.makedirs(args.out, exist_ok=True)


@torch.no_grad()
def decode_svd(x0, chunk=2):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i + chunk], z[i:i + chunk].shape[0]).sample for i in range(0, z.shape[0], chunk)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return v.unsqueeze(0).permute(0, 2, 1, 3, 4).float().cpu()   # [1,3,F,512,512]


@torch.no_grad()
def render(dsl, tag):
    M.set_reference(ref_lat, clip, 1)
    x0, _ = self_forcing_rollout(M, noise, clip, motion, dsl, block_size=8, grad_window=None, full_steps=True)
    print(f"  [{tag}] x0.std={x0.float().std():.3f}")
    return decode_svd(x0)


vids = {}
print("[render] ode-baseline 4step ...")
vids["ode4"] = render([999, 749, 499, 249], "ode-baseline-4step")
if args.dmd2_ckpt:
    rk = torch.load(args.dmd2_ckpt, map_location="cpu")
    sd = rk.get("generator_ema") or rk["generator"]   # 部署用 EMA（全量 state，bf16），无则用 raw
    M.generator.load_state_dict(sd, strict=False)
    print(f"[loaded {'EMA' if 'generator_ema' in rk else 'raw'} dmd2 @ step {rk['step']}]")
    print("[render] dmd2 4step ..."); vids["dmd2_4"] = render([999, 749, 499, 249], "dmd2-4step")
    print("[render] dmd2 1step ..."); vids["dmd2_1"] = render([999], "dmd2-1step")
    both = torch.cat([vids["ode4"], vids["dmd2_4"], vids["dmd2_1"]], dim=4)  # [ode4 | dmd2_4 | dmd2_1]
    save_videos_grid(both, f"{args.out}/{n}_ode4_dmd4_dmd1.mp4", n_rows=1, fps=25)
    for f in [4, 24]:
        if f < F_:
            row = Image.new("RGB", (512 * 3, 512))
            for j, k in enumerate(["ode4", "dmd2_4", "dmd2_1"]):
                arr = (vids[k][0, :, f].permute(1, 2, 0).numpy() * 255).astype("uint8")
                row.paste(Image.fromarray(arr), (512 * j, 0))
            row.save(f"{args.out}/{n}_frame{f}_ode4_dmd4_dmd1.png")
            print(f"  frame{f} (ode4|dmd2_4|dmd2_1) -> {args.out}/{n}_frame{f}_ode4_dmd4_dmd1.png")
    print(f"[done] -> {args.out}/{n}_ode4_dmd4_dmd1.mp4")
else:
    save_videos_grid(vids["ode4"], f"{args.out}/{n}_ode4_only.mp4", n_rows=1, fps=25)
    print(f"[done smoke] ode baseline -> {args.out}/{n}_ode4_only.mp4")

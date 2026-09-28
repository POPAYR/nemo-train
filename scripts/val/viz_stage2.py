"""Stage2 可视化:同样本同噪声,对比 GT / 仅stage1(无temporal) / stage2各checkpoint / 少步。
「仅stage1」那一行是关键对照 —— 它没有 temporal,直接展示 temporal 模块到底贡献了什么。
输出 mp4 + 帧条到 output/viz_stage2/<sample>/。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/viz_stage2.py
"""
import sys, os, argparse, subprocess
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_stage2"
S1 = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_stage1/stage1_step_3000.pt"

# (标签, ckpt, 是否含temporal, 步数)
VARIANTS = [
    ("1_stage1only_noTemporal_N35", S1, False, 35),
    ("2_stage2_ck2000_N35", "output/flow_stage2/stage2_step_2000.pt", True, 35),
    ("3_stage2_ck3000_N35", "output/flow_stage2/stage2_step_3000.pt", True, 35),
    ("4_stage2_ck3000_N8", "output/flow_stage2/stage2_step_3000.pt", True, 8),
    ("5_stage2_ck3000_N4", "output/flow_stage2/stage2_step_3000.pt", True, 4),
]

ap = argparse.ArgumentParser()
ap.add_argument("--samples", nargs="+",
                default=["M003_video_front_happy_level_3_001", "M023_video_right_30_fear_level_3_010"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
_SP = torch.load(cfg.denoising_unet_path, map_location="cpu")


def build(ckpt, with_temporal):
    uak = OmegaConf.to_container(ic.unet_additional_kwargs, resolve=True)
    uak["use_temporal_module"] = with_temporal
    u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=uak).to(device=dev, dtype=dt)
    u.load_state_dict(_SP, strict=False)
    rk = torch.load(ckpt, map_location="cpu"); sd = rk.get("denoising_unet", rk)
    if not with_temporal:
        sd = {k: v for k, v in sd.items() if "temporal_modules" not in k}
    u.load_state_dict(sd, strict=False)
    if with_temporal and rk.get("rope", False):
        set_temporal_rope(u, True, mode="bidir")
    return u.eval(), bool(rk.get("rope", False))


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(g[2:] - 2*g[1:-1] + g[:-2]).mean()
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()
    return d1, d2, d2/d1, hf


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


def save_strip(arr, path, n=8):
    idx = np.linspace(0, len(arr)-1, n).astype(int)
    H, W = arr.shape[1], arr.shape[2]
    row = Image.new("RGB", (W*len(idx), H))
    for j, i in enumerate(idx): row.paste(Image.fromarray(arr[i]), (W*j, 0))
    row.save(path)


for name in a.samples:
    F_ = a.frames
    od = os.path.join(OUT_ROOT, name); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    ref_pil.save(f"{od}/ref.png")
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    gt_lat = _all[:F_].to(dev, dt).permute(1,0,2,3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float().reshape(-1,32*16)[:F_].reshape(1,F_,32,16).to(dev, dt)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    noise = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)

    print(f"\n===== {name}  {F_}帧 seed={a.seed} =====")
    print(f"{'配置':>30} | {'运动':>7} {'抖动':>7} {'抖动比':>7} {'高频':>6}")
    print("-" * 66)
    ga = decode(gt_lat); d1,d2,r,hf = metrics(ga)
    save_mp4(ga, f"{od}/0_GT.mp4"); save_strip(ga, f"{od}/0_GT_strip.png")
    print(f"{'0_GT':>30} | {d1:>7.3f} {d2:>7.3f} {r:>7.3f} {hf:>6.2f}")

    for tag, ck, wt, N in VARIANTS:
        if not os.path.exists(ck): print(f"  [skip] 缺 {ck}"); continue
        u, rope = build(ck, wt)
        rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
        rr = ReferenceAttentionControl(u, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
        rw.clear(); refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
        rr.update(rw, dtype=dt)
        sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0); sch.set_timesteps(N, device=dev)
        z = noise.clone()
        with torch.no_grad():
            for t in sch.timesteps:
                v = u(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
                z = sch.step(v.float(), t, z.float()).prev_sample.to(dt)
        arr = decode(z); d1,d2,r,hf = metrics(arr)
        save_mp4(arr, f"{od}/{tag}.mp4"); save_strip(arr, f"{od}/{tag}_strip.png")
        print(f"{tag:>30} | {d1:>7.3f} {d2:>7.3f} {r:>7.3f} {hf:>6.2f}")
        del u; torch.cuda.empty_cache()

print(f"\n→ {OUT_ROOT}/<sample>/")
print("DONE", flush=True)

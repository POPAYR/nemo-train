"""2 步 vs 4 步 因果流式推理速度基准(fps)。

测的是**真实部署路径**:stream 模式 + KV cache,逐 block 生成,每块 N 步去噪 + 1 次干净 commit。
分三段计时,便于看瓶颈在哪:
  · UNet 生成(潜空间)      —— 步数直接影响这段
  · VAE 解码(潜码→像素)    —— 与步数无关的固定开销
  · 端到端                 —— 部署实际能跑到的 fps

⚠️ 每块的前向次数 = N(去噪) + 1(commit),不是 N。所以 4→2 步的理论加速是
   5/3 = 1.67×,不是 2×。这点在解读结果时别搞错。
rolling KV window:0=无界历史(短片段用);>0=有界显存(长流用,且 compile 需要静态 shape)。

用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/bench_causal_steps.py \
        --ckpt output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt
"""
import sys, os, time, argparse
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
GRIDS = {4: [1.0, 0.75, 0.5, 0.25, 0.0], 2: [1.0, 0.5, 0.0], 1: [1.0, 0.0]}

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt")
ap.add_argument("--key", default="generator_ema")
ap.add_argument("--sample", default="M003_video_front_happy_level_3_001")
ap.add_argument("--steps", type=int, nargs="+", default=[4, 2])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--vae_chunk", type=int, default=2)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
_k = torch.load(a.ckpt, map_location="cpu"); u.load_state_dict(_k[a.key], strict=False); u.eval()
ctrl = TemporalCausalControl(u, block_size=a.block, window=a.window); ctrl.set_rope(True)
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(u, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

n = a.sample; F_ = a.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt")
              .pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
_all = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()
ref_lat = _all[0:1].to(dev, dt)
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float() \
     .reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
print(f"[ckpt] {a.ckpt} key={a.key}   GPU={torch.cuda.get_device_name(0)}", flush=True)
print(f"[cfg ] {F_}帧 block={a.block} window={a.window} vae_chunk={a.vae_chunk}\n", flush=True)


@torch.no_grad()
def gen_latent(N):
    S = GRIDS[N]
    rw.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rr.update(rw, dtype=dt)
    ctrl.set_mode("stream"); ctrl.reset_cache()
    out = []
    for st in range(0, F_, a.block):
        nb = min(a.block, F_ - st)
        mo_b = mo[:, st:st + nb]
        z = torch.randn(1, 4, nb, 64, 64, device=dev, dtype=dt)
        for i in range(N):
            ctrl.set_offset(st); ctrl.set_commit(False)
            v = u(z, torch.full((1, nb), S[i] * 1000.0, device=dev, dtype=dt),
                  encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)[0]
            z = (z.float() + (S[i + 1] - S[i]) * v.float()).to(dt)
        ctrl.set_offset(st); ctrl.set_commit(True)
        u(z, torch.zeros((1, nb), device=dev, dtype=dt),
          encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)
        out.append(z)
    ctrl.set_mode("off"); ctrl.reset_cache()
    return torch.cat(out, dim=2)


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i + a.vae_chunk], z[i:i + a.vae_chunk].shape[0]).sample
            for i in range(0, z.shape[0], a.vae_chunk)]
    return torch.cat(outs, 0)


res = {}
for N in a.steps:
    for _ in range(a.warmup):
        decode(gen_latent(N))
    torch.cuda.synchronize()
    tg, td = [], []
    for _ in range(a.reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lat = gen_latent(N)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        decode(lat)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        tg.append(t1 - t0); td.append(t2 - t1)
    g, d = float(np.median(tg)), float(np.median(td))
    res[N] = (g, d, g + d)
    nfe = (N + 1) * (F_ // a.block)                       # 每块 N 次去噪 + 1 次 commit
    print(f"N={N}步  UNet {g*1000:>7.1f}ms ({F_/g:>6.1f} fps) | "
          f"VAE {d*1000:>7.1f}ms ({F_/d:>6.1f} fps) | "
          f"端到端 {(g+d)*1000:>7.1f}ms (**{F_/(g+d):>5.1f} fps**)  UNet前向={nfe}次", flush=True)

if len(res) > 1:
    ks = sorted(res, reverse=True)
    base, cur = res[ks[0]], res[ks[-1]]
    print(f"\n{ks[-1]}步 vs {ks[0]}步 加速:UNet {base[0]/cur[0]:.2f}×  端到端 {base[2]/cur[2]:.2f}×")
    print(f"(UNet 理论上限 {((ks[0]+1)/(ks[-1]+1)):.2f}× —— 每块含 1 次 commit 前向,故非 {ks[0]/ks[-1]:.0f}×)")
    print(f"VAE 解码是与步数无关的固定开销,占端到端 {cur[1]/cur[2]*100:.0f}%(2步时)")
print("DONE", flush=True)

"""因果 flow 采样器:验证 Phase 2.5 因果 init(block-causal N步 Euler,stream 模式 + KV cache)。
用法: CUDA_VISIBLE_DEVICES=2 python scripts/val/render_flow_causal.py --ckpt output/ode_init_flow/flow_causal_step_2000.pt --steps 4 8"""
import sys, os, argparse, torch, numpy as np
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
from src.distill.flow_math import flow_add_noise, v_to_x0, flow_step_list

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT = "/tmp/claude-1020/-media-ps-ssd5-ayr/1d369635-237e-4560-8f40-baea66749f53/scratchpad/dbg"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sample", default="M023_video_right_30_fear_level_3_010")
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--window", type=int, default=0, help="rolling KV 窗(帧);0=无界历史。训练L=16→部署用~16避免OOD")
ap.add_argument("--tf_history", action="store_true",
                help="诊断用:commit GT 干净帧当历史(=推理也 teacher forcing)。干净则证明残留漂移纯来自自预测累积→SF-DMD 可修")
ap.add_argument("--steps", type=int, nargs="+", default=[4, 8])
args = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
rk = torch.load(args.ckpt, map_location="cpu")
# 兼容三种 ckpt:因果init(denoising_unet) / DMD(generator_ema 优先,回落 generator)
_sd = rk.get("denoising_unet") or rk.get("generator_ema") or rk.get("generator")
_which = "denoising_unet" if "denoising_unet" in rk else ("generator_ema" if rk.get("generator_ema") else "generator")
denu.load_state_dict(_sd, strict=False)
denu.eval()
print(f"[loaded {args.ckpt} step={rk.get('step')} obj={rk.get('objective')} key={_which}]", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()

n = args.sample; F_ = args.frames; BL = args.block
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
_all_lat = torch.load(f"{ROOT}/frame_latent/{n}.pt", map_location="cpu").float()
ref_lat = _all_lat[0:1].to(dev, dt)
# GT 全序列 latent [1,C,F,H,W](--tf_history 诊断用)
gt_lat = _all_lat[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0) if _all_lat.dim() == 4 else None
mo = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float().reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)

ctrl = TemporalCausalControl(denu, block_size=BL, window=args.window)
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i + 2], z[i:i + 2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def denoise(z, sig, mot_b):
    t_emb = torch.full((z.shape[0],), float(sig) * 1000.0, device=dev, dtype=dt)
    v = denu(z, t_emb, encoder_hidden_states=[clip, mot_b], pose_cond_fea=None, return_dict=False)[0]
    return v_to_x0(z, v, torch.tensor(float(sig), device=dev))


@torch.no_grad()
def sample(N):
    rwriter.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    _ts, _sg = flow_step_list(N, shift=1.0)
    sigma_list = [float(x) for x in _sg[:-1].tolist()]              # N 个起始 σ
    ctrl.set_mode("stream"); ctrl.reset_cache()
    noise = torch.randn(1, 4, F_, 64, 64, device=dev, dtype=dt)
    outs = []
    for cur in range(0, F_, BL):
        z = noise[:, :, cur:cur + BL]; mot_b = mo[:, cur:cur + BL]
        ctrl.set_offset(cur)
        x0 = None
        for i, sig in enumerate(sigma_list):
            ctrl.set_commit(False)
            x0 = denoise(z, sig, mot_b)
            if i == len(sigma_list) - 1: break
            z = flow_add_noise(x0, torch.randn_like(x0), sigma_list[i + 1]).to(dt)
        outs.append(x0)
        ctrl.set_commit(True)                                       # 干净 KV commit
        # 默认 commit 模型自己的预测(=SF 推理);--tf_history 改 commit GT(=推理也 teacher forcing,诊断用)
        commit_src = gt_lat[:, :, cur:cur + BL] if (args.tf_history and gt_lat is not None) else x0
        denoise(commit_src, 0.0, mot_b)
    ctrl.set_commit(True); ctrl.set_mode("off")
    x0_full = torch.cat(outs, dim=2)
    arr = decode(x0_full)
    for f in [x for x in [0, 8, 15, 16, 24, 31, 32, 40, 48, min(63, F_ - 1)] if x < F_]:
        Image.fromarray(arr[f]).save(f"{OUT}/causal_N{N}_f{f:02d}.png")
    g = arr.astype(np.float32).mean(3); d = np.abs(np.diff(g, axis=0)).mean()
    if args.tf_history and gt_lat is not None: print(f"  [tf_history] gt_lat.std={gt_lat.float().std():.3f}", flush=True)
    print(f"[causal N={N}] x0.std={x0_full.float().std():.3f} frame-diff={d:.3f} 帧存 causal_N{N}_f*.png", flush=True)


for N in args.steps:
    sample(N)
print("DONE", flush=True)

"""推理侧 guidance 扫描:能否靠 CFG 权重把 DMD 学生的"过锐"压回 GT 水平?

背景:v1_step2000 的时序指标已很好(运动 +7.7% / 抖动 +6.8% / 抖动比 −0.8%),
唯一短板是**空间高频 +21.3%(过锐)**。若推理侧一个标量就能压回去,DMD 阶段即可收工。

⚠️ 学生是 4 步因果、推理**本来不跑 CFG**(引导已烘焙进 ODE 轨迹)。所以这里扫的是
   v = v_uncond + w·(v_cond − v_uncond):
     w = 1.0  → 恰好等于 v_cond,即当前无 CFG 行为(**实现自检点**,应复现原指标)
     w < 1.0  → 向 uncond 插值,降锐度(我们想要的方向)
     w > 1.0  → 更锐
uncond 用 XNeMo 三重置空(bank 不注入 + CLIP 置零 + 参考帧 motion),与训练/渲染一致。
batch=1 时 cat 成 batch=2,uc_mask 正好切对(前半 uncond)——不会踩帧维那个坑。

输出 → output/eval/guidance_sweep/
用法: CUDA_VISIBLE_DEVICES=6 python scripts/val/sweep_causal_guidance.py \
        --ckpt output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt --ws 0.6 0.8 1.0 1.3
"""
import sys, os, argparse, subprocess
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
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/guidance_sweep"
SIGMAS = [1.0, 0.75, 0.5, 0.25, 0.0]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt")
ap.add_argument("--key", default="generator_ema")
ap.add_argument("--ws", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.3])
ap.add_argument("--samples", nargs="+",
                default=["M028_video_front_surprised_level_1_004",
                         "M011_video_front_angry_level_3_013",
                         "W011_video_top_happy_level_2_011",
                         "W019_video_left_60_surprised_level_2_023"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
os.makedirs(OUT_ROOT, exist_ok=True)
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
CKTAG = os.path.basename(a.ckpt).replace(".pt", "")

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()

u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
_k = torch.load(a.ckpt, map_location="cpu")
u.load_state_dict(_k[a.key], strict=False); u.eval()
ctrl = TemporalCausalControl(u, block_size=a.block, window=0)
ctrl.set_rope(True)
print(f"[ckpt] {a.ckpt} key={a.key} step={_k.get('step')}", flush=True)

# CFG:writer/reader 都开 do_cfg → uncond(前半 batch)在 attn1 里走纯自注意力(bank 不注入)
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=True, mode="write", batch_size=1, fusion_blocks="full")
rr = ReferenceAttentionControl(u, do_classifier_free_guidance=True, mode="read", batch_size=1, fusion_blocks="full")


def metrics(arr):
    g = arr.astype(np.float32).mean(3)
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean()
    hf = np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()
    return d1, d2, d2 / d1, hf


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def sample(ref_lat, clip, motion, F_, w, gen):
    """4 步因果流式 + CFG 权重 w。全程 batch=2(uncond 在前半)。"""
    clip_cat = torch.cat([torch.zeros_like(clip), clip], 0)
    mo_neg = motion[:, 0:1].expand_as(motion).contiguous()
    rw.clear()
    refu(ref_lat.repeat(2, 1, 1, 1), torch.zeros((), device=dev).long(),
         encoder_hidden_states=clip_cat, return_dict=False)
    rr.update(rw, dtype=dt)
    ctrl.set_mode("stream"); ctrl.reset_cache()
    out = []
    for st in range(0, F_, a.block):
        nb = min(a.block, F_ - st)
        mo_c = torch.cat([mo_neg[:, st:st + nb], motion[:, st:st + nb]], 0)
        z = torch.randn(1, 4, nb, 64, 64, generator=gen, device=dev, dtype=dt)
        for i in range(4):
            s_cur, s_nxt = SIGMAS[i], SIGMAS[i + 1]
            ctrl.set_offset(st); ctrl.set_commit(False)
            v2 = u(torch.cat([z, z], 0), torch.full((2, nb), s_cur * 1000.0, device=dev, dtype=dt),
                   encoder_hidden_states=[clip_cat, mo_c], pose_cond_fea=None, return_dict=False)[0]
            vu, vc = v2.float().chunk(2)
            v = vu + w * (vc - vu)
            z = (z.float() + (s_nxt - s_cur) * v).to(dt)
        ctrl.set_offset(st); ctrl.set_commit(True)          # 两个半 batch 提交同一份干净块
        u(torch.cat([z, z], 0), torch.zeros((2, nb), device=dev, dtype=dt),
          encoder_hidden_states=[clip_cat, mo_c], pose_cond_fea=None, return_dict=False)
        out.append(z)
    ctrl.set_mode("off"); ctrl.reset_cache()
    return torch.cat(out, dim=2)


agg = {w: [] for w in a.ws}; agg_gt = []
for name in a.samples:
    F_ = a.frames
    od = os.path.join(OUT_ROOT, name); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt")
                  .pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    gt_lat = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float() \
         .reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)

    print(f"\n===== {name} =====")
    print(f"{'配置':>14} | {'运动':>7} {'抖动':>7} {'抖动比':>7} {'高频':>6}")
    ga = decode(gt_lat); m = metrics(ga); agg_gt.append(m)
    save_mp4(ga, f"{od}/00_GT.mp4")
    print(f"{'GT':>14} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)
    for w in a.ws:
        g = torch.Generator(device=dev); g.manual_seed(a.seed)
        arr = decode(sample(ref_lat, clip, mo, F_, w, g)); m = metrics(arr); agg[w].append(m)
        save_mp4(arr, f"{od}/{CKTAG}_w{w}.mp4")
        print(f"{'w='+str(w):>14} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)

gt = np.array(agg_gt).mean(0)
print(f"\n===== {len(a.samples)} 样本均值(相对 GT %)=====")
print(f"{'':<10}{'运动':>10}{'抖动':>10}{'抖动比':>10}{'高频':>10}{'归一MAE':>10}")
print(f"{'GT':<10}"+"".join(f"{v:>10.3f}" for v in gt))
for w in a.ws:
    v = np.array(agg[w]); rel = (v.mean(0) - gt) / gt * 100
    mae = (np.abs(v - np.array(agg_gt)).mean(0) / gt).sum()
    print(f"{'w='+str(w):<10}"+"".join(f"{x:>+10.1f}" for x in rel)+f"{mae:>10.4f}")
print("DONE", flush=True)

"""Stage1 图像模型评测:结构健康度 + 单帧画质 + **采样步数扫描(flow vs ε 对照)**。
stage1 无 temporal,故是纯粹的「参考图 + motion token → 目标帧」任务,且**目标帧已知**
→ 可以直接算重建指标(L1 / PSNR),这是视频阶段做不到的。

步数扫描回答:rectified flow 到底能把采样步数压到多少,以及相对原始 ε+DDIM 的优势有多大。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/eval_stage1_image.py --ckpt output/flow_stage1/stage1_step_1000.pt
"""
import sys, os, argparse, random
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import (AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler, DDIMScheduler)
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.data.image_pair_dataset import ImagePairDataset

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/viz_stage1"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n_pair", type=int, default=8, help="评测多少组 (参考帧, 目标帧)")
ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 35])
ap.add_argument("--min_gap", type=int, default=20, help="参考-目标最小间隔,保证是真正的跨帧重建")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_eps", action="store_true", help="跳过 ε 基线")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
os.makedirs(OUT, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()


def build_unet(no_temporal, ckpt=None):
    uak = OmegaConf.to_container(ic.unet_additional_kwargs, resolve=True)
    uak["use_temporal_module"] = not (not no_temporal)   # no_temporal=True → False
    uak["use_temporal_module"] = False if no_temporal else True
    u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=uak).to(device=dev, dtype=dt)
    u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    if not no_temporal:
        u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("denoising_unet", sd)
        u.load_state_dict({k: v for k, v in sd.items() if "temporal_modules" not in k}, strict=False)
    return u.eval()


# ---- 取评测对(与训练同源但固定 seed;n_target=1) ----
dcfg = OmegaConf.load(TRAIN_CFG).data
vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
src = dcfg.sources[1] if len(dcfg.sources) > 1 else dcfg.sources[0]
ds = ImagePairDataset(pose_dir=src.pose_dir, latent_dir=src.latent_dir, video_dir=src.video_dir,
                      data_name_path=src.data_name_path, n_target=1, min_gap=a.min_gap, video_processor=vproc)
pairs = [ds[random.randrange(len(ds))] for _ in range(a.n_pair)]
print(f"取 {a.n_pair} 组跨帧对,参考-目标间隔 = {[int(p['gap'][0]) for p in pairs]}\n")

ref_lat = torch.stack([p["ref_latent"] for p in pairs]).to(dev, dt)
ref_img = torch.stack([p["ref_img"] for p in pairs]).to(dev, dt)
gt_lat = torch.stack([p["tgt_latent"][0] for p in pairs]).to(dev, dt)          # [P,C,H,W]
motion = torch.stack([p["tgt_motion"] for p in pairs]).to(dev, dt)            # [P,1,32,16]
P = len(pairs)


@torch.no_grad()
def decode(lat):                       # [P,C,H,W] → uint8 [P,H,W,3]
    z = lat / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def metrics(pred, gt):
    p = pred.astype(np.float32); g = gt.astype(np.float32)
    l1 = np.abs(p - g).mean()
    psnr = 10 * np.log10(255.0 ** 2 / max(((p - g) ** 2).mean(), 1e-8))
    gray = p.mean(3)
    hf = np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=2)).mean()
    return l1, psnr, hf


clip_emb = imgenc(ref_img).image_embeds.unsqueeze(1)
gt_arr = decode(gt_lat)
_, _, gt_hf = metrics(gt_arr, gt_arr)
print(f"GT 空间高频(锐度基准) = {gt_hf:.2f}\n")


def setref(u):
    rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=P, fusion_blocks="full")
    rr = ReferenceAttentionControl(u, do_classifier_free_guidance=False, mode="read", batch_size=P, fusion_blocks="full")
    rw.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip_emb, return_dict=False)
    rr.update(rw, dtype=dt)


@torch.no_grad()
def sample_flow(u, N):
    setref(u)
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    sch.set_timesteps(N, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    z = torch.randn(P, 4, 1, 64, 64, generator=g, device=dev, dtype=dt)
    for t in sch.timesteps:
        v = u(z, t.expand(P).to(dt), encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
        z = sch.step(v.float(), t, z.float()).prev_sample.to(dt)
    return z[:, :, 0]


@torch.no_grad()
def sample_eps(u, N):
    setref(u)
    sch = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
    sch.set_timesteps(N, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    z = torch.randn(P, 4, 1, 64, 64, generator=g, device=dev, dtype=dt) * sch.init_noise_sigma
    for t in sch.timesteps:
        e = u(z, t, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
        z = sch.step(e.float(), t, z.float()).prev_sample.to(dt)
    return z[:, :, 0]


rows = []
print(f"{'模型':>16} {'步数':>5} | {'L1↓':>7} {'PSNR↓dB↑':>9} {'空间高频':>9} {'vs GT锐度':>10} {'x0.std':>7}")
print("-" * 74)
u_flow = build_unet(no_temporal=True, ckpt=a.ckpt)
for N in a.steps:
    x0 = sample_flow(u_flow, N); arr = decode(x0)
    l1, ps, hf = metrics(arr, gt_arr)
    print(f"{'flow(stage1)':>16} {N:>5} | {l1:>7.2f} {ps:>9.2f} {hf:>9.2f} {hf/gt_hf*100:>9.0f}% {x0.float().std():>7.3f}")
    rows.append(("flow", N, arr))
    del x0
torch.cuda.empty_cache()

if not a.no_eps:
    u_eps = build_unet(no_temporal=True, ckpt=None)      # 原始 ε 空间权重,同样无 temporal
    for N in a.steps:
        x0 = sample_eps(u_eps, N); arr = decode(x0)
        l1, ps, hf = metrics(arr, gt_arr)
        print(f"{'ε原始(DDIM)':>16} {N:>5} | {l1:>7.2f} {ps:>9.2f} {hf:>9.2f} {hf/gt_hf*100:>9.0f}% {x0.float().std():>7.3f}")
        rows.append(("eps", N, arr))
        del x0

# ---- 拼图:每行一个配置,列=各评测对 ----
def save_grid(tag, items):
    H = W = 512
    cols = min(P, 6)
    rowsn = len(items) + 2                     # +ref +gt
    canvas = Image.new("RGB", (W * cols, H * rowsn))
    ref_arr = decode(ref_lat)
    for j in range(cols): canvas.paste(Image.fromarray(ref_arr[j]), (W * j, 0))
    for j in range(cols): canvas.paste(Image.fromarray(gt_arr[j]), (W * j, H))
    for i, (mdl, N, arr) in enumerate(items):
        for j in range(cols): canvas.paste(Image.fromarray(arr[j]), (W * j, H * (i + 2)))
    p = f"{OUT}/{tag}.jpg"; canvas.save(p, quality=92)
    print(f"\n拼图 → {p}  (第1行=参考帧, 第2行=GT目标帧, 其后依次: " +
          ", ".join(f"{m}-N{n}" for m, n, _ in items) + ")")


save_grid(f"stage1_steps_{os.path.basename(a.ckpt).replace('.pt','')}", rows)
print("DONE", flush=True)

"""滑窗渲染 vs 一次性渲染 —— 复刻 XNeMo 正统的 24帧窗/4帧重叠。
动机:ε baseline 用 context_frames=24/overlap=4 的滑窗,**每个去噪步内**把各窗口的预测在重叠区平均
(pipeline_pose2vid_motenc_long.py:612),这是很强的方差抑制 → 静止背景被"钉住"。
我们此前一律 64 帧一次性,单次采样方差原封不动留在输出里 → 观感上像"摄像机在动"。
本脚本给 flow 模型也上滑窗,同条件对比。

正统细节(全部复刻):
  - 每个去噪步重新随机 offset(random.randint(0, context_frames-1)),窗口边界在步间抖动
  - 平均在**预测值 v** 上做,再做 CFG,最后 scheduler.step
  - CFG 用三重置空 uncond(bank 不注入 + 零CLIP + 参考帧motion),见 xnemo-cfg 记忆
输出 → output/eval/window_compare/(用户可见目录)
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/render_window_compare.py --ckpt output/flow_stage2/stage2_step_4000.pt
"""
import sys, os, argparse, subprocess, random
import torch, numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference/scripts/val")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope
from src.pipelines.context import get_context_scheduler

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/window_compare"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="output/flow_stage2/stage2_step_4000.pt")
ap.add_argument("--samples", nargs="+",
                default=["M003_video_front_happy_level_3_001", "M023_video_right_30_fear_level_3_010"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--context_frames", type=int, default=24)
ap.add_argument("--context_overlap", type=int, default=4)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
_rk = torch.load(a.ckpt, map_location="cpu")
denu.load_state_dict(_rk["denoising_unet"], strict=False); denu.eval()
if bool(_rk.get("rope", False)): set_temporal_rope(denu, True, mode="bidir")
CKTAG = os.path.basename(a.ckpt).replace(".pt", "")
print(f"[ckpt] {a.ckpt} step={_rk.get('step')} rope={_rk.get('rope')}", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=True, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=True, mode="read", batch_size=1, fusion_blocks="full")
ctx_sched = get_context_scheduler("uniform")


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


def save_mp4(arr, path):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); p.wait()


def regional(arr):
    g = arr.astype(np.float32).mean(3)
    H, W = g.shape[1], g.shape[2]
    yy, xx = np.mgrid[0:H, 0:W]
    face = ((yy-H/2)**2/(0.32*H)**2 + (xx-W/2)**2/(0.32*W)**2) < 1.0
    bg = ~face
    tstd = g.std(axis=0); d1 = np.abs(np.diff(g, axis=0)).mean(axis=0)
    gx = np.abs(np.diff(g, axis=2)); gy = np.abs(np.diff(g, axis=1))
    sh = np.zeros_like(g[0]); sh[:, :-1] += gx.mean(axis=0); sh[:-1, :] += gy.mean(axis=0)
    return sh[bg].mean(), sh[face].mean(), tstd[bg].mean(), tstd[face].mean(), d1[bg].mean()


@torch.no_grad()
def sample(z0, clip, clip_cat, mo, mo_neg, ref_lat, N, cs, use_window):
    F_ = z0.shape[2]
    rwriter.clear()
    refu(ref_lat.repeat(2, 1, 1, 1), torch.zeros((), device=dev).long(),
         encoder_hidden_states=clip_cat, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    sch.set_timesteps(N, device=dev)
    z = z0.clone()
    for t in sch.timesteps:
        te2 = t.expand(2).to(dt)
        if not use_window:
            v2 = denu(torch.cat([z, z], 0), te2,
                      encoder_hidden_states=[clip_cat, torch.cat([mo_neg, mo], 0)],
                      pose_cond_fea=None, return_dict=False)[0].float()
        else:
            # ★ 每步重新随机 offset;窗口预测在重叠区累加求平均(与原 pipeline 一致)
            off = random.randint(0, a.context_frames - 1)
            queue = list(ctx_sched(0, N, F_, a.context_frames, 1, a.context_overlap, True, off))
            acc = torch.zeros(2, *z.shape[1:], device=dev, dtype=torch.float32)
            cnt = torch.zeros(2, 1, F_, 1, 1, device=dev, dtype=torch.float32)
            for c in queue:
                zc = torch.cat([z[:, :, c], z[:, :, c]], 0)
                mc = torch.cat([mo_neg[:, c], mo[:, c]], 0)
                pv = denu(zc, te2, encoder_hidden_states=[clip_cat, mc],
                          pose_cond_fea=None, return_dict=False)[0].float()
                acc[:, :, c] += pv
                cnt[:, :, c] += 1
            v2 = acc / cnt.clamp_min(1)
        vu, vc = v2.chunk(2)
        v = vu + cs * (vc - vu)
        z = sch.step(v, t, z.float()).prev_sample.to(dt)
    return z


for name in a.samples:
    F_ = a.frames
    od = os.path.join(OUT_ROOT, name); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    ref_pil.save(f"{od}/ref.png")
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224,224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    clip_cat = torch.cat([torch.zeros_like(clip), clip], 0)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    gt_lat = _all[:F_].to(dev, dt).permute(1,0,2,3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float().reshape(-1,32*16)[:F_].reshape(1,F_,32,16).to(dev, dt)
    mo_neg = mo[:, 0:1].expand_as(mo).contiguous()
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    z0 = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)

    print(f"\n===== {name}  {F_}帧 N={a.steps} cfg={a.cfg} =====")
    print(f"{'配置':>34} | {'背景锐度':>8} {'人脸锐度':>8} | {'背景时序std':>11} {'人脸时序std':>11} {'背景帧差':>8}")
    print("-" * 96)
    ga = decode(gt_lat); m = regional(ga)
    save_mp4(ga, f"{od}/00_GT.mp4")
    print(f"{'00_GT':>34} | {m[0]:>8.3f} {m[1]:>8.3f} | {m[2]:>11.3f} {m[3]:>11.3f} {m[4]:>8.3f}")

    for uw, tag in [(False, "onepass_64f"), (True, f"window{a.context_frames}ov{a.context_overlap}")]:
        random.seed(a.seed)
        x0 = sample(z0, clip, clip_cat, mo, mo_neg, ref_lat, a.steps, a.cfg, uw)
        arr = decode(x0); m = regional(arr)
        fn = f"ours_{CKTAG}_N{a.steps}_cfg{a.cfg}_{tag}.mp4"
        save_mp4(arr, f"{od}/{fn}")
        print(f"{tag:>34} | {m[0]:>8.3f} {m[1]:>8.3f} | {m[2]:>11.3f} {m[3]:>11.3f} {m[4]:>8.3f}")

print(f"\n→ {OUT_ROOT}/<sample>/")
print("DONE", flush=True)

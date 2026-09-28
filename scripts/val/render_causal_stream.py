"""因果 init 的真正验收:**流式 4 步推理**(stream 模式 + KV cache,逐 block 生成)。

loss 下降说明不了因果 init 成功 —— 训练是并行 block-causal mask 的一次前向,
而推理是逐块流式、历史来自自己已 commit 的干净 KV。真正会塌的地方在后者。
本脚本三路对照,把"少步代价"和"因果代价"分开:
  1. GT(VAE 往返)
  2. 双向 CUM1500 @ 4 步   —— 只有少步代价
  3. 因果 ckpt @ 4 步流式   —— 少步 + 因果代价

σ 网格用 [1.0, 0.75, 0.5, 0.25](= 官方 denoising_step_list/1000),**不用** diffusers 的
set_timesteps(4)(末档 σ=0.001 与终点重合,第 4 步空转)。与 gen_ode_pairs.py 训练时一致。
不做 CFG:引导已烘焙进 ODE 轨迹,学生学的就是已引导的映射。

输出 → output/eval/causal_init/(用户可见目录)
用法: CUDA_VISIBLE_DEVICES=1 python scripts/val/render_causal_stream.py \
        --ckpt output/ode_init_causal/odeinit_step_500.pt
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
from src.models.temporal_causal import TemporalCausalControl, set_temporal_rope

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/causal_init"
SIGMAS = [1.0, 0.75, 0.5, 0.25, 0.0]        # 4 步 + 终点

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--bidir", default="output/flow_stage2_cfgdrop/CUM1500.pt")
ap.add_argument("--samples", nargs="+",
                default=["M003_video_front_happy_level_3_001",
                         "W019_video_left_60_surprised_level_2_023"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--window", type=int, default=0, help="rolling KV 窗(帧);0=无界历史")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--no_bidir", action="store_true",
                help="跳过双向 4 步基线(它是常量,多 ckpt 横比时跑一遍就够)")
ap.add_argument("--sigmas", type=float, nargs="+", default=None,
                help="自定义 σ 网格(不含末尾0),如 2步用 --sigmas 1.0 0.5。"
                     "⭐ 4步轨迹的 KEEP_IDX=[0,3,6,9,12] 已含 σ=1.0/0.5,故 2 步无需重新生成 ODE pairs")
ap.add_argument("--key", default="auto",
                help="ckpt 里取哪套权重:auto/denoising_unet(causal init)/generator_ema(DMD部署权重)/generator")
a = ap.parse_args()
if a.sigmas:
    SIGMAS = list(a.sigmas) + [0.0]
NSTEP = len(SIGMAS) - 1
dev = torch.device("cuda:0"); dt = torch.bfloat16
os.makedirs(OUT_ROOT, exist_ok=True)
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
CKTAG = os.path.basename(a.ckpt).replace(".pt", "") + ("_ema" if a.key in ("auto","generator_ema") else "")

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()


def build_unet(ckpt, key="auto"):
    u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    k = torch.load(ckpt, map_location="cpu")
    if key == "auto":
        # DMD ckpt 优先取 EMA(部署权重);causal init / stage2 ckpt 用 denoising_unet
        key = next(x for x in ("generator_ema", "denoising_unet", "generator") if x in k)
    sd = k[key]
    print(f"  [load] {os.path.basename(ckpt)} ← key='{key}' ({len(sd)} 项)", flush=True)
    u.load_state_dict(sd, strict=False)
    return u.eval(), k, key


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


def save_strip(arr, path, n=8):
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    Image.fromarray(np.concatenate([arr[i] for i in idx], axis=1)).save(path)


@torch.no_grad()
def decode(x0):
    z = rearrange(x0, "b c f h w -> (b f) c h w") / 0.18215
    outs = [vae.decode(z[i:i+2], z[i:i+2].shape[0]).sample for i in range(0, z.shape[0], 2)]
    v = (torch.cat(outs, 0) / 2 + 0.5).clamp(0, 1)
    return (v.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def sample_bidir(u, rr, rw, ref_lat, clip, motion, F_, gen):
    """双向 4 步:一次性整段,只含少步代价。"""
    rw.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rr.update(rw, dtype=dt)
    z = torch.randn(1, 4, F_, 64, 64, generator=gen, device=dev, dtype=dt)
    for i in range(NSTEP):
        s_cur, s_nxt = SIGMAS[i], SIGMAS[i + 1]
        v = u(z, torch.full((1, F_), s_cur * 1000.0, device=dev, dtype=dt),
              encoder_hidden_states=[clip, motion], pose_cond_fea=None, return_dict=False)[0]
        z = (z.float() + (s_nxt - s_cur) * v.float()).to(dt)
    return z


@torch.no_grad()
def sample_causal_stream(u, ctrl, rr, rw, ref_lat, clip, motion, F_, gen):
    """因果 4 步流式:逐 block 去噪,块干净后 commit 进 KV,下一块只看历史。"""
    rw.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rr.update(rw, dtype=dt)
    ctrl.set_mode("stream"); ctrl.reset_cache()
    out = []
    for st in range(0, F_, a.block):
        nb = min(a.block, F_ - st)
        mo_b = motion[:, st:st + nb]
        z = torch.randn(1, 4, nb, 64, 64, generator=gen, device=dev, dtype=dt)
        for i in range(NSTEP):
            s_cur, s_nxt = SIGMAS[i], SIGMAS[i + 1]
            ctrl.set_offset(st); ctrl.set_commit(False)      # 中间步只读 KV,不写
            v = u(z, torch.full((1, nb), s_cur * 1000.0, device=dev, dtype=dt),
                  encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)[0]
            z = (z.float() + (s_nxt - s_cur) * v.float()).to(dt)
        ctrl.set_offset(st); ctrl.set_commit(True)           # 用干净块写一次 KV
        u(z, torch.zeros((1, nb), device=dev, dtype=dt),
          encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)
        out.append(z)
    ctrl.set_mode("off"); ctrl.reset_cache()
    return torch.cat(out, dim=2)


ub, kb, _ = build_unet(a.bidir, "denoising_unet")
if bool(kb.get("rope", False)):
    set_temporal_rope(ub, True, mode="bidir")
uc, kc, _ck_key = build_unet(a.ckpt, a.key)
ctrl = TemporalCausalControl(uc, block_size=a.block, window=a.window)
ctrl.set_rope(True)
print(f"[bidir] {a.bidir}  [causal] {a.ckpt} step={kc.get('step')} key={_ck_key}", flush=True)

rw_b = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr_b = ReferenceAttentionControl(ub, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
rr_c = ReferenceAttentionControl(uc, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

for name in a.samples:
    F_ = a.frames
    od = os.path.join(OUT_ROOT, name); os.makedirs(od, exist_ok=True)
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    ref_pil.save(f"{od}/ref.png")
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt")
                  .pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    gt_lat = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float() \
         .reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)

    print(f"\n===== {name}  {F_}帧 block={a.block} σ={SIGMAS[:-1]} =====")
    print(f"{'配置':>40} | {'运动':>7} {'抖动':>7} {'抖动比':>7} {'高频':>6}")
    print("-" * 80)
    ga = decode(gt_lat); m = metrics(ga)
    save_mp4(ga, f"{od}/00_GT.mp4"); save_strip(ga, f"{od}/00_GT_strip.png")
    print(f"{'00_GT':>40} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)

    runs = [] if a.no_bidir else [(f"bidir_CUM1500_{NSTEP}step",
                                   lambda g: sample_bidir(ub, rr_b, rw_b, ref_lat, clip, mo, F_, g))]
    runs.append((f"causal_{CKTAG}_{NSTEP}step_stream",
                 lambda g: sample_causal_stream(uc, ctrl, rr_c, rw_b, ref_lat, clip, mo, F_, g)))
    for tag, fn_ in runs:
        g = torch.Generator(device=dev); g.manual_seed(a.seed)
        arr = decode(fn_(g)); m = metrics(arr)
        save_mp4(arr, f"{od}/{tag}.mp4"); save_strip(arr, f"{od}/{tag}_strip.png")
        print(f"{tag:>40} | {m[0]:>7.3f} {m[1]:>7.3f} {m[2]:>7.3f} {m[3]:>6.2f}", flush=True)

print(f"\n→ {OUT_ROOT}/<sample>/")
print("DONE", flush=True)

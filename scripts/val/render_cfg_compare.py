"""CFG 对比渲染 —— 输出到 output/eval/cfg_compare/(用户可见工作目录,非 /tmp)。
把 GT、我们的 stage2 各 CFG、以及 ε 正规配置(cfg2.5+24帧滑窗)放在一起,文件名自解释。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/render_cfg_compare.py \
        --ckpt output/flow_stage2/stage2_step_4000.pt --cfgs 1.0 2.0 2.5 3.0
"""
import sys, os, argparse, subprocess
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

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/cfg_compare"   # ★ 可见工作目录

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="output/flow_stage2/stage2_step_4000.pt")
ap.add_argument("--samples", nargs="+",
                default=["M003_video_front_happy_level_3_001", "M023_video_right_30_fear_level_3_010"])
ap.add_argument("--frames", type=int, default=64)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--cfgs", type=float, nargs="+", default=[1.0, 2.0, 2.5, 3.0])
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--cfg_mode", choices=["motion", "full"], default="full",
                help="full=**正统 XNeMo 语义**(scripts/pose2vid_xnemo.py → pipeline_pose2vid_motenc_long):"
                     "uncond 三重置空 = reference bank 不注入(mutual_self_attention.py:194 走纯自注意力)"
                     " + CLIP 置零 + 参考帧 motion,需 2x batch;"
                     "motion=早期错误近似(bank/CLIP 都保留,只换 motion),仅作对照")
ap.add_argument("--with_eps_native", action="store_true", default=True,
                help="同时渲染 ε 正规配置(cfg2.5 + 24帧滑窗)做对照")
ap.add_argument("--no_eps", dest="with_eps_native", action="store_false",
                help="跳过 ε 基线(多样本选 ckpt 时基线是常量,省一半时间)")
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
_rope = bool(_rk.get("rope", False))
if _rope: set_temporal_rope(denu, True, mode="bidir")
CKTAG = f"{os.path.basename(a.ckpt).replace('.pt','')}"
print(f"[ckpt] {a.ckpt} step={_rk.get('step')} stage={_rk.get('stage')} rope={_rope}", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
# ★ 两套 reference control:cfg=1 必须用 do_cfg=False 的那套。
#   否则 uc_mask 在 batch=1 时按 hidden_states.shape[0]//2 切分 —— 切的是**帧维**,
#   导致前一半帧完全不注入 reference bank(表现为"视频前半段崩")。
rwriter_n = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader_n = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
rwriter_c = ReferenceAttentionControl(refu, do_classifier_free_guidance=True, mode="write", batch_size=1, fusion_blocks="full")
rreader_c = ReferenceAttentionControl(denu, do_classifier_free_guidance=True, mode="read", batch_size=1, fusion_blocks="full")
print(f"[cfg_mode] {a.cfg_mode}  (cfg=1 用 do_cfg=False 那套 control,避免 uc_mask 切帧维的坑)", flush=True)


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
    mo_neg = mo[:, 0:1].expand_as(mo).contiguous()

    print(f"\n===== {name}  {F_}帧 N={a.steps} seed={a.seed} =====")
    print(f"{'文件':>44} | {'运动':>7} {'抖动':>7} {'抖动比':>7} {'高频':>6}")
    print("-" * 82)
    ga = decode(gt_lat); d1,d2,r,hf = metrics(ga)
    fn = "00_GT.mp4"; save_mp4(ga, f"{od}/{fn}"); save_strip(ga, f"{od}/00_GT_strip.png")
    print(f"{fn:>44} | {d1:>7.3f} {d2:>7.3f} {r:>7.3f} {hf:>6.2f}")

    clip_cat = torch.cat([torch.zeros_like(clip), clip], dim=0)      # [uncond(零CLIP), cond]
    mo_cat = torch.cat([mo_neg, mo], dim=0)
    for cs in a.cfgs:
        use_cfg = cs > 1.0
        full = use_cfg and a.cfg_mode == "full"
        rw = rwriter_c if full else rwriter_n
        rr = rreader_c if full else rreader_n
        rw.clear()
        with torch.no_grad():
            if full:
                # ★ 正统:reference UNet 跑 2 遍 —— uncond 用零 CLIP,产生「无外观提示」的 bank;
                #   且 ReferenceAttentionControl(do_cfg=True) 会让前半 batch 完全不注入 bank。
                refu(ref_lat.repeat(2, 1, 1, 1), torch.zeros((), device=dev).long(),
                     encoder_hidden_states=clip_cat, return_dict=False)
            else:
                refu(ref_lat, torch.zeros((), device=dev).long(),
                     encoder_hidden_states=clip, return_dict=False)
            rr.update(rw, dtype=dt)
            sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
            sch.set_timesteps(a.steps, device=dev)
            g = torch.Generator(device=dev); g.manual_seed(a.seed)
            z = torch.randn(1, 4, F_, 64, 64, generator=g, device=dev, dtype=dt)
            for t in sch.timesteps:
                if full:                                   # uncond 在前半(uc_mask 约定)
                    v2 = denu(torch.cat([z, z], dim=0), t.expand(2).to(dt),
                              encoder_hidden_states=[clip_cat, mo_cat],
                              pose_cond_fea=None, return_dict=False)[0]
                    vu, vc = v2.float().chunk(2)
                    v = vu + cs * (vc - vu)
                elif use_cfg:                              # motion-only 近似(对照)
                    vc = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo],
                              pose_cond_fea=None, return_dict=False)[0]
                    vu = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo_neg],
                              pose_cond_fea=None, return_dict=False)[0]
                    v = vu.float() + cs * (vc.float() - vu.float())
                else:
                    v = denu(z, t.expand(1).to(dt), encoder_hidden_states=[clip, mo],
                             pose_cond_fea=None, return_dict=False)[0].float()
                z = sch.step(v, t, z.float()).prev_sample.to(dt)
        arr = decode(z); d1,d2,r,hf = metrics(arr)
        fn = f"ours_{CKTAG}_N{a.steps}_cfg{cs}_{a.cfg_mode}.mp4"
        save_mp4(arr, f"{od}/{fn}"); save_strip(arr, f"{od}/{fn.replace('.mp4','_strip.png')}")
        print(f"{fn:>44} | {d1:>7.3f} {d2:>7.3f} {r:>7.3f} {hf:>6.2f}")

    if a.with_eps_native:
        try:
            from test_ar_model import load_xnemo_pipeline, render
            import argparse as _ap
            pipe = load_xnemo_pipeline(cfg, dev, torch.float16)
            mo2 = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float().reshape(-1,32*16)[:F_]
            g = torch.Generator(device=dev); g.manual_seed(a.seed)
            ns = _ap.Namespace(W=512,H=512,steps=a.steps,cfg=2.5,context_frames=24,context_overlap=4)
            vid = render(pipe, ref_pil, ref_pil, mo2.unsqueeze(0).to(dev), ns, g)
            arr = (vid[0].permute(1,2,3,0).float().cpu().numpy()*255).astype(np.uint8)
            d1,d2,r,hf = metrics(arr)
            fn = "baseline_epsDDPM_N35_cfg2.5_win24.mp4"
            save_mp4(arr, f"{od}/{fn}"); save_strip(arr, f"{od}/{fn.replace('.mp4','_strip.png')}")
            print(f"{fn:>44} | {d1:>7.3f} {d2:>7.3f} {r:>7.3f} {hf:>6.2f}")
            del pipe; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [skip ε基线] {repr(e)[:120]}")

print(f"\n→ 全部输出在 {OUT_ROOT}/<sample>/  (mp4 + _strip.png + ref.png)")
print("DONE", flush=True)

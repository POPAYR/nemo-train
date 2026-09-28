"""诊断:flow teacher 在各 σ 上的预测精度 —— 检验低噪声区是否欠训。
良好训练的扩散/flow 模型,x0 预测误差应随 σ **单调下降**(噪声越小越容易)。
若低 σ 处误差不降反平/上翘,说明该区域训练密度不足(我们用 logit-normal σ,P(σ<0.1)仅 1.42%)。
对照:同一批数据上,把每个 σ 的 x0-MSE 归一化到「相对该 σ 的理论难度」。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/diag_sigma_coverage.py --ckpt output/flow_teacher/flow_teacher_FINAL.pt
"""
import sys, argparse, torch
import numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.distill.flow_math import flow_add_noise, v_to_x0

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/media/ps/ssd5/ayr/x-nemo-inference/output/flow_teacher/flow_teacher_FINAL.pt")
ap.add_argument("--samples", nargs="+", default=["M023_video_right_30_fear_level_3_010",
                                                 "M003_video_front_happy_level_3_001"])
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--reps", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
rk = torch.load(a.ckpt, map_location="cpu")
denu.load_state_dict(rk["denoising_unet"], strict=False); denu.eval()
print(f"[loaded] {a.ckpt} step={rk.get('step')}", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

SIGMAS = [0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.7, 0.9]
acc = {s: [] for s in SIGMAS}
acc_v = {s: [] for s in SIGMAS}

for name in a.samples:
    F_ = a.frames
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    _all = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = _all[0:1].to(dev, dt)
    x0 = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
    with torch.no_grad():
        rwriter.clear()
        refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
        rreader.update(rwriter, dtype=dt)
        for sg in SIGMAS:
            for _ in range(a.reps):
                sigma = torch.full((1,), sg, device=dev)
                n = torch.randn_like(x0)
                z = flow_add_noise(x0, n, sigma).to(dt)
                te = torch.full((1,), sg * 1000.0, device=dev, dtype=dt)
                v = denu(z, te, encoder_hidden_states=[clip, mo], pose_cond_fea=None, return_dict=False)[0]
                xp = v_to_x0(z, v, sigma)
                acc[sg].append(((xp.float() - x0.float()) ** 2).mean().item())
                acc_v[sg].append(((v.float() - (n.float() - x0.float())) ** 2).mean().item())

print(f"\n{'σ':>6} | {'x0-MSE':>9} {'单调?':>6} | {'v-MSE':>8} | {'训练密度':>9}")
print("-" * 56)
prev = None
dens_ln = {0.03: 0.06, 0.05: 0.17, 0.08: 0.8, 0.12: 2.4, 0.2: 8.4, 0.3: 20, 0.5: 50, 0.7: 80, 0.9: 98}
for sg in SIGMAS:
    m0 = float(np.mean(acc[sg])); mv = float(np.mean(acc_v[sg]))
    flag = "" if prev is None else ("✓" if m0 < prev else "✗上翘")
    print(f"{sg:>6.2f} | {m0:>9.4f} {flag:>6} | {mv:>8.4f} | {dens_ln[sg]:>8.1f}%")
    prev = m0
print("\n判读:x0-MSE 应随 σ 减小而单调下降;低 σ 处若下降乏力/上翘 → 该区欠训。")
print("      「训练密度」= logit-normal 下 P(σ < 该值) 的累积占比。")
print("DONE", flush=True)

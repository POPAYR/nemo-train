"""
Phase-0 后续: (A) 干净部署态推理显存  (B) TAESD vs SVD-decoder 验证+计时
=====================================================================
A) 部署态显存: denoising UNet(1.69B)+reference UNet(0.86B)+CLIP+VAE, fp16,
   跑一次 K=8 去噪 forward + VAE decode, 报 weights / peak。回答 "24G 能否推理"。
B) TAESD: 用 SVD encoder 把真实人脸帧 -> latent (SD 4ch), 分别用 SVD temporal decoder
   和 TAESD 解码, 比 PSNR + 存并排图 + 计时 ms/帧。回答 "TAESD 画质/速度够不够"。

用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/bench_mem_taesd.py
"""
import sys, os, numpy as np, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderTiny
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl

dev = torch.device("cuda:0"); dt = torch.float16
torch.backends.cudnn.benchmark = True
cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")
ic = OmegaConf.load(cfg.inference_config)
GB = 1024**3


def med_ms(fn, wmp=3, it=10):
    for _ in range(wmp): fn()
    torch.cuda.synchronize(); xs = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); xs.append(s.elapsed_time(e))
    return float(np.median(xs))


print("="*70); print("PART A — 部署态推理显存"); print("="*70)
torch.cuda.reset_peak_memory_stats()
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt).eval()
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
clipenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
w_mem = torch.cuda.memory_allocated() / GB
print(f"[weights] denoising+reference+VAE(SVD)+CLIP fp16 常驻 = {w_mem:.2f} GB")

w = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
r = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
clip = torch.randn(1, 1, 768, device=dev, dtype=dt); refl = torch.randn(1, 4, 64, 64, device=dev, dtype=dt); t = torch.tensor(500, device=dev)
w.clear(); refu(refl, torch.zeros_like(t), encoder_hidden_states=clip, return_dict=False); r.update(w)
K = 8; lat = torch.randn(1, 4, K, 64, 64, device=dev, dtype=dt); mot = torch.randn(1, K, 32, 16, device=dev, dtype=dt)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    denu(lat, t, encoder_hidden_states=[clip, mot], pose_cond_fea=None, return_dict=False)
    z = torch.randn(K, 4, 64, 64, device=dev, dtype=dt) / 0.18215
    vae.decode(z, K)
peak = torch.cuda.max_memory_allocated() / GB
print(f"[peak] K=8 去噪 forward + SVD VAE decode 峰值 = {peak:.2f} GB")
print(f"  -> 换 TAESD 后 VAE 常驻从 ~{sum(p.numel() for p in vae.parameters())*2/GB:.2f}GB 降到 ~0.01GB, decode 峰值也更小")

print("\n" + "="*70); print("PART B — TAESD vs SVD decoder 验证"); print("="*70)
# 真实人脸帧
img_path = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps/face_frames/W011_video_right_30_fear_level_3_007/000036.jpg"
if not os.path.exists(img_path):
    img_path = "/media/ps/ssd5/ayr/x-nemo-inference/demo/ref_image.png"
pil = Image.open(img_path).convert("RGB").resize((512, 512))
x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0).to(dev, dt)  # [-1,1]

with torch.no_grad():
    z_raw = vae.encode(x).latent_dist.mean          # SD 4ch raw latent
    z_scaled = z_raw * 0.18215                       # UNet 空间 (×scaling)
    # SVD decode (基准): 喂 raw latent
    svd = vae.decode(z_raw, 1).sample                # [-1,1] [1,3,512,512]

taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dt).to(dev).eval()
print(f"[TAESD] params={sum(p.numel() for p in taesd.parameters())/1e6:.2f}M  scaling_factor={taesd.config.scaling_factor}")

def to01(t): return (t.float()/2+0.5).clamp(0,1)
def psnr(a, b):
    mse = torch.mean((to01(a)-to01(b))**2).item()
    return 10*np.log10(1.0/max(mse,1e-10))

# TAESD 期望的 latent 缩放未知 → 试几种, 取与原图 PSNR 最高的
cands = {"raw(z)": z_raw, "scaled(z*0.18215)": z_scaled, "raw*0.5": z_raw*0.5}
best = None
with torch.no_grad():
    for name, zz in cands.items():
        try:
            dec = taesd.decode(zz).sample
            p = psnr(dec, x); p_vs_svd = psnr(dec, svd)
            print(f"  TAESD[{name:20s}] PSNR vs原图={p:5.2f}dB  vs SVD={p_vs_svd:5.2f}dB")
            if best is None or p > best[1]: best = (name, p, dec)
        except Exception as ex:
            print(f"  TAESD[{name}] err {ex}")
svd_psnr = psnr(svd, x)
print(f"  SVD-decoder        PSNR vs原图={svd_psnr:5.2f}dB  (基准上界)")
print(f"  -> TAESD 最优缩放 = '{best[0]}'  PSNR={best[1]:.2f}dB  (vs SVD {svd_psnr:.2f}dB, 差 {svd_psnr-best[1]:.2f}dB)")

# 存并排图: 原图 | SVD | TAESD
outdir = "/media/ps/ssd5/ayr/x-nemo-inference/output/taesd_check"; os.makedirs(outdir, exist_ok=True)
def t2pil(t): return Image.fromarray((to01(t)[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
cat = Image.new("RGB", (512*3, 512))
cat.paste(pil, (0,0)); cat.paste(t2pil(svd), (512,0)); cat.paste(t2pil(best[2]), (1024,0))
cat.save(os.path.join(outdir, "orig_svd_taesd.png"))
print(f"  并排图(原图|SVD|TAESD) -> {outdir}/orig_svd_taesd.png")

# 计时 K=8 decode
zk = z_raw.repeat(8,1,1,1)
zk_taesd = (best[2] is not None) and cands[best[0]].repeat(8,1,1,1)
t_svd = med_ms(lambda: vae.decode(zk, 8))
t_tae = med_ms(lambda: taesd.decode(cands[best[0]].repeat(8,1,1,1)))
print(f"\n[计时 K=8 decode]  SVD={t_svd:.1f}ms ({t_svd/8:.2f}ms/帧)  |  TAESD={t_tae:.1f}ms ({t_tae/8:.2f}ms/帧)  加速 {t_svd/max(t_tae,1e-6):.0f}x")

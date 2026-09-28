"""在**同一批训练数据、同一 σ 序列**上比较各 teacher 的 flow v-MSE —— 建立可比的 loss 基准。
背景:RoPE run 的 loss 卡在 ~0.32,而拿来对比的 0.25 是「train_scope=all + L=40 + 加性PE」时测的,
三个变量同时不同,不可比。本脚本固定数据与噪声,只换权重/PE,给出真正的 apples-to-apples。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/diag_train_loss_parity.py --L 64 --n 24
"""
import sys, argparse
import torch
import numpy as np
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"

CKPTS = [
    ("加性PE teacher(FINAL)", "output/flow_teacher/flow_teacher_FINAL.pt"),
    ("RoPE teacher step1000", "output/flow_teacher_rope/flow_step_1000.pt"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--n", type=int, default=24, help="评测 clip 数")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
_SP = torch.load(cfg.denoising_unet_path, map_location="cpu")
_TP = torch.load(cfg.temporal_module_path, map_location="cpu")
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")

dcfg = OmegaConf.load(TRAIN_CFG).data
vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
ds = ConcatDataset([MotarDataset(pose_dir=s.pose_dir, audio_dir=s.audio_dir, caption_dir=s.caption_dir,
        data_name_path=s.data_name_path, tokenizer_path=dcfg.tokenizer_path, data_stats_path=dcfg.data_stats_path,
        context_length=a.L, fps=dcfg.fps, sr=dcfg.sr, text_max_len=dcfg.get("text_max_len", 128),
        random_crop=True, pad_short=True, load_video=True, latent_dir=s.latent_dir,
        video_dir=s.video_dir, video_processor=vproc) for s in dcfg.sources])
g = torch.Generator(); g.manual_seed(a.seed)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, generator=g)
stats = torch.load(dcfg.data_stats_path, map_location="cpu")
m_mean = stats["mean"].reshape(-1).to(dev, dt); m_std = stats["std"].reshape(-1).to(dev, dt)

# 固定取 n 个 clip + 固定 σ/噪声,所有模型共用
batches = []
it = iter(dl)
torch.manual_seed(a.seed)
for i in range(a.n):
    b = next(it)
    sig = torch.sigmoid(torch.randn(1, device=dev)).clamp(1e-3, 1 - 1e-3)
    batches.append((b, sig, torch.Generator(device=dev).manual_seed(1000 + i)))
print(f"固定 {a.n} 个 clip、L={a.L}、σ 与噪声全部固定,只换权重\n")


@torch.no_grad()
def evaluate(ckpt, use_rope):
    denu.load_state_dict(_SP, strict=False); denu.load_state_dict(_TP, strict=False)
    rk = torch.load(ckpt, map_location="cpu")
    denu.load_state_dict(rk["denoising_unet"], strict=False); denu.eval()
    set_temporal_rope(denu, use_rope, mode="bidir" if use_rope else "off")
    tot, per_sig = 0.0, []
    for b, sig, gen in batches:
        x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()
        B, T = x0.shape[0], x0.shape[2]
        motion = (b["motion_tensor"].to(dev, dt) * (m_std + 1e-6) + m_mean).reshape(B, T, 32, 16)
        ref_lat = b["ref_latent"].to(dev, dt)
        clip = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
        rwriter.clear()
        refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
        rreader.update(rwriter, dtype=dt)
        noise = torch.randn(x0.shape, generator=gen, device=dev, dtype=dt)
        sb = sig.view(1, 1, 1, 1, 1)
        z = ((1 - sb) * x0.float() + sb * noise.float()).to(dt)
        vt = (noise.float() - x0.float())
        te = (sig * 1000.0).to(dt)
        vp = denu(z, te, encoder_hidden_states=[clip, motion], pose_cond_fea=None, return_dict=False)[0]
        l = ((vp.float() - vt) ** 2).mean().item()
        tot += l / len(batches); per_sig.append((sig.item(), l))
    return tot, per_sig


print(f"{'模型':>24} | {'flow v-MSE':>11} | {'低σ(<0.4)':>10} {'中σ':>8} {'高σ(>0.6)':>10}")
print("-" * 72)
res = {}
for name, ck in CKPTS:
    use_rope = bool(torch.load(ck, map_location="cpu").get("rope", False))
    tot, ps = evaluate(ck, use_rope)
    lo = np.mean([l for s, l in ps if s < 0.4]) if any(s < 0.4 for s, _ in ps) else float("nan")
    mid = np.mean([l for s, l in ps if 0.4 <= s <= 0.6]) if any(0.4 <= s <= 0.6 for s, _ in ps) else float("nan")
    hi = np.mean([l for s, l in ps if s > 0.6]) if any(s > 0.6 for s, _ in ps) else float("nan")
    print(f"{name:>24} | {tot:>11.4f} | {lo:>10.4f} {mid:>8.4f} {hi:>10.4f}   (rope={use_rope})")
    res[name] = tot
if len(res) == 2:
    k = list(res)
    d = res[k[1]] - res[k[0]]
    print(f"\n差值 = {d:+.4f}  ({'RoPE 更差' if d > 0 else 'RoPE 更好'},相对 {abs(d)/res[k[0]]*100:.1f}%)")
print("DONE", flush=True)

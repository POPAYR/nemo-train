"""分段 CFG 的阈值诊断:CFG 在哪个 σ 以上开始把 DMD 目标推离真实分布?

背景(1.x-Distill, arXiv 2604.04018 式4):DMD 的 mode collapse 部分源于
**teacher 在高噪声段用了过强的 CFG** —— 结构在高噪段形成,强引导把结构钉死在主模态。
其修法是 timestep-aware CFG:t≤α 开 CFG,t>α 完全关闭(α=0.94,w=7.0,shift=3.0)。

⚠️ 不照抄 α=0.94:
  · 他们 shift=3.0,我们 shift=1.0 —— 同一个 t 对应的实际噪声不同(σ=s·t/(1+(s-1)t))
  · 论文中 t 是 shift 前还是后,表述有歧义
  · 我们 DMD 打分的 σ 只采到 0.98,阈值若在 0.94 只命中约 4% 样本,近乎空操作
故本脚本**实测**我们自己的 teacher:逐 σ 量 CFG 把 x0_real 推向还是推离真实数据。

判据(以 GT 潜码为真实分布的样本):
  d_cond = ‖x0_cond − x0_gt‖ ,  d_cfg = ‖x0_cfg − x0_gt‖
  d_cfg > d_cond  → 该 σ 上 CFG 在**推离**真实数据 → 应关闭
另报 std 与 ‖Δ‖=‖x0_cfg − x0_cond‖(CFG 改动幅度),看代价与收益是否匹配。

同时测两种输入分布(DMD 打分实际作用在后者):
  [GT]  z 由真实潜码加噪 —— 干净参照
  [GEN] z 由学生 rollout 的 x0 加噪 —— 训练时的真实工况

用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/diag_cfg_vs_sigma.py \
        --teacher output/flow_stage2_cfgdrop/CUM1500.pt \
        --student output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt --w 2.5
"""
import sys, os, argparse
import numpy as np
import torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl, set_temporal_rope

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"
OUT = "/media/ps/ssd5/ayr/x-nemo-inference/output/eval/cfg_sigma_diag"

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="output/flow_stage2_cfgdrop/CUM1500.pt")
ap.add_argument("--student", default="output/flowdmd_causal_v1/ckpt/dmd2_step_2000.pt")
ap.add_argument("--w", type=float, default=2.5,
                help="我们的公式是 cond + w·(cond−unc) → 等效 CFG = w+1")
ap.add_argument("--sigmas", type=float, nargs="+",
                default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
ap.add_argument("--samples", nargs="+",
                default=["M003_video_front_happy_level_3_001",
                         "W019_video_left_60_surprised_level_2_023",
                         "M011_video_front_angry_level_3_013"])
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--reps", type=int, default=3, help="每个 σ 重复(不同噪声)取均值")
ap.add_argument("--block", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16
os.makedirs(OUT, exist_ok=True)
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()


def build(ckpt, key, causal):
    u = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    k = torch.load(ckpt, map_location="cpu")
    u.load_state_dict(k[key], strict=False); u.eval()
    c = None
    if causal:
        c = TemporalCausalControl(u, block_size=a.block, window=0); c.set_rope(True)
    else:
        set_temporal_rope(u, True, mode="bidir")
    return u, c


teacher, _ = build(a.teacher, "denoising_unet", causal=False)
student, sctrl = build(a.student, "generator_ema", causal=True)
rw = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
rr_t = ReferenceAttentionControl(teacher, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
rr_s = ReferenceAttentionControl(student, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
print(f"[teacher] {a.teacher}\n[student] {a.student}\n[w] {a.w} (等效 CFG {a.w+1.0})", flush=True)


@torch.no_grad()
def set_ref(ref_lat, clip):
    rw.clear()
    refu(ref_lat, torch.zeros((), device=dev).long(), encoder_hidden_states=clip, return_dict=False)
    rr_t.update(rw, dtype=dt); rr_s.update(rw, dtype=dt)


@torch.no_grad()
def teacher_x0(z, sigma, clip, motion, uncond=False):
    """uncond=True 时走三重置空:bank 不注入 + CLIP 置零 + 参考帧 motion。"""
    F_ = z.shape[2]
    if uncond:
        rr_t.clear()
        ce = torch.zeros_like(clip)
        mo = motion[:, 0:1].expand_as(motion).contiguous()
    else:
        ce, mo = clip, motion
    v = teacher(z, torch.full((1, F_), sigma * 1000.0, device=dev, dtype=dt),
                encoder_hidden_states=[ce, mo], pose_cond_fea=None, return_dict=False)[0]
    if uncond:
        rr_t.update(rw, dtype=dt)                    # 立刻恢复 bank
    return z.float() - sigma * v.float()             # x0 = z − σ·v


@torch.no_grad()
def student_rollout(clip, motion, F_, g):
    """4 步因果流式,给出 DMD 打分实际面对的 x0 分布。"""
    S = [1.0, 0.75, 0.5, 0.25, 0.0]
    sctrl.set_mode("stream"); sctrl.reset_cache()
    out = []
    for st in range(0, F_, a.block):
        nb = min(a.block, F_ - st)
        mo_b = motion[:, st:st + nb]
        z = torch.randn(1, 4, nb, 64, 64, generator=g, device=dev, dtype=dt)
        for i in range(4):
            sctrl.set_offset(st); sctrl.set_commit(False)
            v = student(z, torch.full((1, nb), S[i] * 1000.0, device=dev, dtype=dt),
                        encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)[0]
            z = (z.float() + (S[i + 1] - S[i]) * v.float()).to(dt)
        sctrl.set_offset(st); sctrl.set_commit(True)
        student(z, torch.zeros((1, nb), device=dev, dtype=dt),
                encoder_hidden_states=[clip, mo_b], pose_cond_fea=None, return_dict=False)
        out.append(z)
    sctrl.set_mode("off"); sctrl.reset_cache()
    return torch.cat(out, dim=2)


rows = {"GT": {}, "GEN": {}}
for name in a.samples:
    F_ = a.frames
    ref_pil = Image.open(f"{ROOT}/face_frames/{name}/000000.jpg").convert("RGB").resize((512, 512))
    clip = imgenc(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt")
                  .pixel_values.to(dev, dt)).image_embeds.unsqueeze(1)
    lat = torch.load(f"{ROOT}/frame_latent/{name}.pt", map_location="cpu").float()
    ref_lat = lat[0:1].to(dev, dt)
    x0_gt = lat[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
    mo = torch.load(f"{ROOT}/pose_embed/{name}.pt", map_location="cpu").float() \
         .reshape(-1, 32 * 16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
    set_ref(ref_lat, clip)
    g = torch.Generator(device=dev); g.manual_seed(0)
    x0_gen = student_rollout(clip, mo, F_, g)
    set_ref(ref_lat, clip)                                    # rollout 后重置 bank

    for tag, base in (("GT", x0_gt.float()), ("GEN", x0_gen.float())):
        for s in a.sigmas:
            dc, df, dl, sc, sf = [], [], [], [], []
            for r in range(a.reps):
                n = torch.randn_like(base)
                z = ((1 - s) * base + s * n).to(dt)
                xc = teacher_x0(z, s, clip, mo, uncond=False)
                xu = teacher_x0(z, s, clip, mo, uncond=True)
                xg = xc + a.w * (xc - xu)                     # 我们的公式:cond + w·Δ
                dc.append((xc - x0_gt.float()).pow(2).mean().sqrt().item())
                df.append((xg - x0_gt.float()).pow(2).mean().sqrt().item())
                dl.append((xg - xc).pow(2).mean().sqrt().item())
                sc.append(xc.std().item()); sf.append(xg.std().item())
            rows[tag].setdefault(s, []).append(
                (np.mean(dc), np.mean(df), np.mean(dl), np.mean(sc), np.mean(sf)))

print(f"\n{'':6}{'σ':>6} | {'d_cond':>8} {'d_cfg':>8} {'Δ距离':>8} | {'‖ΔCFG‖':>8} | "
      f"{'std_cond':>9} {'std_cfg':>9} | 判读")
print("-" * 96)
verdict = {}
for tag in ("GT", "GEN"):
    print(f"[{tag}] z 由{'真实潜码' if tag=='GT' else '学生 rollout'}加噪   (GT x0.std="
          f"{'—'}）", flush=True)
    for s in a.sigmas:
        m = np.array(rows[tag][s]).mean(0)
        dc, df, dl, sc, sf = m
        worse = df - dc
        flag = "CFG 推离真实 ✗" if worse > 0 else "CFG 拉近真实 ✓"
        verdict.setdefault(tag, []).append((s, worse))
        print(f"{'':6}{s:>6.2f} | {dc:>8.4f} {df:>8.4f} {worse:>+8.4f} | {dl:>8.4f} | "
              f"{sc:>9.4f} {sf:>9.4f} | {flag}")
    print()

print("=" * 96)
for tag in ("GT", "GEN"):
    bad = [s for s, w in verdict[tag] if w > 0]
    if bad:
        print(f"[{tag}] CFG 有害的 σ 区间起点 ≈ {min(bad):.2f}  (有害档: {[round(x,2) for x in bad]})")
    else:
        print(f"[{tag}] 所有 σ 上 CFG 都在拉近真实分布 —— 分段 CFG 在本 teacher 上无依据")
print("\n※ 阈值应取 GEN 行(DMD 打分的真实工况);若有害区间落在 σ>0.94,")
print("  而我们 DMD 只采到 σ≤0.98,则分段 CFG 命中率过低,需同时考虑 importance timestep sampling。")
print("DONE", flush=True)

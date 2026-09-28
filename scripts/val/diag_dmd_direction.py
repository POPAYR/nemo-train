"""诊断:DMD 梯度到底指向"更锐"还是"更糊"?
对给定 ckpt(含 generator+critic)与冻结 flow teacher:
  1) 用 GT 真实视频 latent 当 x0(排除 generator 自身问题的干扰),也测 generator rollout 的 x0
  2) 在若干 σ 上做 flow 加噪 z=(1-σ)x0+σε
  3) 算 x0_real(teacher)、x0_fake(critic)、grad=(fake-real)/norm、DMD 目标 target=x0-grad
  4) 报告 std(x0) / std(x0_real) / std(x0_fake) / std(target)
判读: std(target) > std(x0) → DMD 在锐化(方向对);< → DMD 在变糊(方向错)。
用法: CUDA_VISIBLE_DEVICES=1 python scripts/val/diag_dmd_direction.py --ckpt output/flowdmd_v1/ckpt/dmd2_step_500.pt
"""
import sys, argparse, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor
from src.distill.models import DMD2Models
from src.distill.flow_math import flow_add_noise
from src.distill.dmd_loss import dmd_kl_grad

FLOW_TEACHER = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_stage2_cfgdrop/CUM1500.pt"
ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="DMD ckpt(含 generator/critic)")
ap.add_argument("--sample", default="M023_video_right_30_fear_level_3_010")
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.95])
ap.add_argument("--reps", type=int, default=4, help="每个 σ 重复次数(不同噪声)取均值")
ap.add_argument("--flow_ckpt", default=None,
                help="teacher 权重。⚠️ 默认的 flow_teacher_FINAL 是 train_scope=all 练坏的那个"
                     "(脸有暗斑),在它上面测出的 σ 窗口不能外推。正式 teacher 用 CUM1500。")
a = ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.bfloat16

M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=8, objective="flow", flow_ckpt=(a.flow_ckpt or FLOW_TEACHER))
rk = torch.load(a.ckpt, map_location="cpu")
M.critic.load_state_dict(rk["critic"], strict=True)
gsd = rk.get("generator_ema") or rk["generator"]
M.generator.load_state_dict(gsd, strict=False)
M.critic.eval(); M.generator.eval()
print(f"[loaded] {a.ckpt} step={rk.get('step')}  critic+generator 已载入,teacher={a.flow_ckpt or FLOW_TEACHER}", flush=True)

F_ = a.frames
_all = torch.load(f"{ROOT}/frame_latent/{a.sample}.pt", map_location="cpu").float()
ref_lat = _all[0:1].to(dev, dt)
gt = _all[:F_].to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)          # [1,C,F,H,W]
mo = torch.load(f"{ROOT}/pose_embed/{a.sample}.pt", map_location="cpu").float().reshape(-1, 32*16)[:F_].reshape(1, F_, 32, 16).to(dev, dt)
ref_pil = Image.open(f"{ROOT}/face_frames/{a.sample}/000000.jpg").convert("RGB").resize((512, 512))
clip = M.clip_embed(CLIPImageProcessor().preprocess(ref_pil.resize((224, 224)), return_tensors="pt").pixel_values.to(dev, dt))
M.set_reference(ref_lat, clip, 1)

print(f"\n{'σ':>6} | {'std(x0)':>8} {'std(real)':>9} {'std(fake)':>9} | {'std(target)':>11} {'Δ vs x0':>9} | 判读")
print("-" * 78)


@torch.no_grad()
def probe(x0, tag):
    print(f"[{tag}]  x0.std={x0.float().std():.3f}")
    for sg in a.sigmas:
        acc = {k: 0.0 for k in ("real", "fake", "tgt")}
        for _ in range(a.reps):
            sigma = torch.full((1,), sg, device=dev)
            n = torch.randn_like(x0)
            z = flow_add_noise(x0, n, sigma).to(dt)
            _, x0_real = M.forward_net_flow(M.teacher, z, sigma, clip, mo)
            _, x0_fake = M.forward_net_flow(M.critic, z, sigma, clip, mo)
            grad = dmd_kl_grad(x0, x0_fake, x0_real)
            tgt = x0 - grad
            acc["real"] += x0_real.float().std().item() / a.reps
            acc["fake"] += x0_fake.float().std().item() / a.reps
            acc["tgt"] += tgt.float().std().item() / a.reps
        s0 = x0.float().std().item()
        d = acc["tgt"] - s0
        verdict = "锐化 ✓" if d > 0.005 else ("变糊 ✗" if d < -0.005 else "中性")
        print(f"{sg:>6.2f} | {s0:>8.3f} {acc['real']:>9.3f} {acc['fake']:>9.3f} | "
              f"{acc['tgt']:>11.3f} {d:>+9.3f} | {verdict}")


probe(gt, "GT 真实视频 latent")
print()
# generator 自己 rollout 的 x0(4 步完整)
from src.distill.flow_rollout import flow_rollout
from src.distill.flow_math import flow_step_list
_ts, _sg = flow_step_list(4, shift=1.0)
sigma_list = [float(x) for x in _sg[:-1].tolist()]
with torch.no_grad():
    noise = torch.randn(1, 4, F_, 64, 64, device=dev, dtype=dt)
    x0_gen, _ = flow_rollout(M, noise, clip, mo, sigma_list, block_size=8, grad_window=None, full_steps=True)
probe(x0_gen, "generator rollout x0")
print("\nDONE", flush=True)

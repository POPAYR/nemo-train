"""
DMD-through-G 真实 plumbing 冒烟（修正版：渲染器=已蒸4步因果解码器，打分=未蒸teacher单次前向）
=====================================================================================
- 渲染器 R = DMD2Models.generator（causal，从 dmd2_win24_0701/step_26000）→ self_forcing_rollout 可微渲 24 帧。
- s_real = DMD2Models.teacher（未蒸 teacher，单次前向打分 | m_gt）。
- s_fake = DMD2Models.critic（单次前向打分 | m̂）。
证明：render(可微)→ DMD → motion.grad + critic 更新 + 显存。
用法：CUDA_VISIBLE_DEVICES=2 python scripts/val/smoke_dmd_through_g.py
"""
import os, sys
import torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from src.distill.models import DMD2Models
from src.distill.rollout import self_forcing_rollout
from src.distill.dmd_through_g import generator_step, critic_step

GEN_CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"
DSL = [999, 749, 499, 249]   # 4 步（对齐验证过的视频 DMD）


def main():
    dev = torch.device("cuda:0"); dt = torch.bfloat16
    print("[load] DMD2Models（渲染器 generator + teacher + critic）...")
    M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=8)
    sd = torch.load(GEN_CKPT, map_location="cpu")
    M.generator.load_state_dict(sd["generator"], strict=True)   # 蒸馏 4 步解码器权重
    print(f"[load] generator ← {os.path.basename(GEN_CKPT)}")

    B, C, F, H, W = 1, 4, 24, 64, 64
    clip_emb = torch.randn(B, 1, 768, device=dev, dtype=dt)
    ref_latent = torch.randn(B, C, H, W, device=dev, dtype=dt)
    M.set_reference(ref_latent, clip_emb, B)

    # dummy motion tokens（m̂ 带梯度；m_gt 无梯度）——模拟 rollout 输出 denorm 后的 24 帧窗
    mot_hat = torch.randn(B, F, 32, 16, device=dev, dtype=dt, requires_grad=True)
    mot_gt = torch.randn(B, F, 32, 16, device=dev, dtype=dt)

    # 渲染器：蒸馏 generator 4 步 causal rollout（full_steps 跑满 + grad_window=F 整窗可微）
    def render_fn():
        noise = torch.randn(B, C, F, H, W, device=dev, dtype=dt)
        x0, _ = self_forcing_rollout(M, noise, clip_emb, mot_hat, DSL,
                                     block_size=8, grad_window=F, full_steps=True)
        return x0
    def s_real_fn(x_t, t): return M.forward_net(M.teacher, x_t, t, clip_emb, mot_gt)[1]
    def s_fake_fn(x_t, t): return M.forward_net(M.critic, x_t, t, clip_emb, mot_hat.detach())[1]
    def s_fake_eps_fn(x_t, t): return M.forward_net(M.critic, x_t, t, clip_emb, mot_hat.detach())[0]

    torch.cuda.reset_peak_memory_stats()
    print("[G] generator_step（蒸馏解码器渲染 → DMD → motion.grad）...")
    l_dmd, x_hat, log = generator_step(render_fn, s_real_fn, s_fake_fn, M.scheduler, dtype=dt)
    l_dmd.backward()
    g = mot_hat.grad.abs().max().item()
    print(f"    L_DMD={l_dmd.item():.4f}  x0std={log['x0_std']:.3f}  motion.grad={g:.3e}  "
          f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB  ({'✅ 梯度回 motion' if g > 0 else '❌'})")

    print("[C] critic_step ×4 ...")
    critic_ps = [p for p in M.critic.parameters() if p.requires_grad]
    critic_opt = torch.optim.AdamW(critic_ps, lr=4e-7, betas=(0.0, 0.999), weight_decay=0.01)
    for it in range(4):
        critic_opt.zero_grad(set_to_none=True)
        l_c, _ = critic_step(x_hat, s_fake_eps_fn, M.scheduler, dtype=dt)
        l_c.backward(); gn = torch.nn.utils.clip_grad_norm_(critic_ps, 10.0)
        critic_opt.step()
        print(f"    it{it}: L_critic(εMSE)={l_c.item():.4f}  gradnorm={gn:.3e}")

    print(f"[mem] 峰值 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("DMD-through-G 真实 plumbing 冒烟完成 ✅")


if __name__ == "__main__":
    main()

"""
Phase 3: DMD2 一步的 generator_loss / critic_loss（DECODER_DISTILL_PLAN.md §Phase 3）
=====================================================================================
- generator_loss：self-forcing rollout(全长 L，随机 24 帧窗带梯度) → x0；**只在该 24 帧窗上**采 t 加噪、
  critic 出 x0_fake、teacher 出 x0_real(可CFG)；DMD grad=(x0_fake−x0_real)/norm；L_G=0.5·MSE(x0_win, sg(...))。
- critic_loss：rollout(no_grad) → x0；**随机 24 帧窗**加噪；critic 预 ε；ε-MSE 去噪 loss。
⭐ 2026-07-01：DMD 打分窗限制在 ≤24 帧 —— teacher 是 24 帧滑窗训的，喂它 64 帧会 OOD、score 不准；
  随机连续窗既让 teacher 待在分布内，又覆盖所有 block 边界；student rollout 仍全长 L（练长流式）。
注：首版 guidance_scale=0（无 CFG）；CFG(neutral motion+zero clip) 作下一步质量精修。
"""
import torch
import torch.nn.functional as F
from .rollout import self_forcing_rollout
from .dmd_loss import (dmd_generator_loss, critic_eps_loss, gan_g_loss, gan_d_loss,
                       gan_g_loss_rel, gan_d_loss_rel, gan_reg_r1r2)

DMD_WINDOW = 24  # teacher 原生 24 帧推理窗（≤PE max 32）→ DMD 打分/梯度只在此长度的随机连续窗


def _add_dmd_noise(M, x0, t_low=20, t_high=980):
    B = x0.shape[0]
    t = torch.randint(t_low, t_high, (B,), device=x0.device)
    n = torch.randn_like(x0)
    x_t = M.scheduler.add_noise(x0, n, t).to(M.dt)
    return x_t, t, n


def generator_loss(M, noise, clip_emb, motion, denoising_step_list,
                   block_size=8, grad_window=DMD_WINDOW, guidance_scale=0.0, gan_g_weight=0.0,
                   real_latent=None, relativistic=False):
    # 1) self-forcing rollout（全长 L；随机连续 grad_window 帧窗保留梯度，其余 no_grad）
    x0, win = self_forcing_rollout(M, noise, clip_emb, motion, denoising_step_list,
                                   block_size=block_size, grad_window=grad_window)
    k, Wn = win
    x0_win = x0[:, :, k:k + Wn]              # 只窗内可微（rollout 已 grad 重叠块）
    mot_win = motion[:, k:k + Wn]
    # 2) KL grad（全 no_grad；teacher/critic 只看 Wn≤24 帧 → PE 0..Wn-1，落在 teacher 分布内）
    with torch.no_grad():
        x_t, t, _ = _add_dmd_noise(M, x0_win.detach())
        _, x0_fake = M.forward_net(M.critic, x_t, t, clip_emb, mot_win)
        _, x0_real = M.forward_net(M.teacher, x_t, t, clip_emb, mot_win)
        if guidance_scale > 0:    # CFG（首版默认关；neg_motion 待 plumb）
            _, x0_unc = M.forward_net(M.teacher, x_t, t, torch.zeros_like(clip_emb), mot_win * 0)
            x0_real = x0_real + guidance_scale * (x0_real - x0_unc)
    # 3) DMD loss（整窗，无需 mask——窗内所有帧都打分）
    loss = dmd_generator_loss(x0_win, x0_fake, x0_real)
    log = {"x0_std": x0.detach().float().std().item()}
    # 4) GAN-G（DMD2 少步锐化）：判别器出 fake logit，grad 经 critic 主干回传到 generator
    #    （critic/disc_head 参数也吃到 grad，但 generator step 只更 gen_opt，其余下轮 zero_grad 清掉）
    if gan_g_weight > 0:
        x_tg, tg, _ = _add_dmd_noise(M, x0_win)      # 不 detach → 梯度回 generator
        _, _, logit_fake = M.forward_net(M.critic, x_tg, tg, clip_emb, mot_win, capture_disc=True)
        if relativistic and real_latent is not None:
            with torch.no_grad():   # real 不依赖 generator → no_grad；同一 tg/同窗
                real_win = real_latent[:, :, k:k + Wn]
                xr = M.scheduler.add_noise(real_win, torch.randn_like(real_win), tg).to(M.dt)
                _, _, logit_real = M.forward_net(M.critic, xr, tg, clip_emb, mot_win, capture_disc=True)
            g_gan = gan_g_loss_rel(logit_fake, logit_real)
        else:
            g_gan = gan_g_loss(logit_fake)
        loss = loss + gan_g_weight * g_gan
        log["gan_g"] = g_gan.detach().item()
    return loss, log


def critic_loss(M, noise, clip_emb, motion, denoising_step_list, block_size=8, grad_window=DMD_WINDOW,
                real_latent=None, gan_d_weight=0.0, relativistic=False, r1r2_weight=0.0, r1r2_sigma=0.05):
    # 1) rollout 出 student 当前样本（no_grad，不训 generator）
    with torch.no_grad():
        x0, _ = self_forcing_rollout(M, noise, clip_emb, motion, denoising_step_list,
                                     block_size=block_size, grad_window=None)
        F_ = x0.shape[2]
        k = int(torch.randint(0, F_ - grad_window + 1, (1,)).item())   # 随机连续 24 帧窗
        x0_win = x0[:, :, k:k + grad_window]
        mot_win = motion[:, k:k + grad_window]
        x_t, t, n = _add_dmd_noise(M, x0_win)
    # 2) critic 预 ε，ε-MSE 去噪 loss（也只在 24 帧窗，分布内）；GAN 时同一前向抓 fake logit
    do_gan = gan_d_weight > 0 and real_latent is not None
    if do_gan:
        eps_pred, _, logit_fake = M.forward_net(M.critic, x_t, t, clip_emb, mot_win, capture_disc=True)
    else:
        eps_pred, _ = M.forward_net(M.critic, x_t, t, clip_emb, mot_win)
    loss = critic_eps_loss(eps_pred, n)
    log = {"critic_x0_std": x0.float().std().item()}
    # 3) GAN-D（DMD2）：real 用同一 t 加噪、同一随机窗，判别器学 real↑/fake↓（real 是数据，no_grad 加噪）
    if do_gan:
        real_win = real_latent[:, :, k:k + grad_window]              # [B,C,F,H,W] 同窗
        with torch.no_grad():
            xr = M.scheduler.add_noise(real_win, torch.randn_like(real_win), t).to(M.dt)
        _, _, logit_real = M.forward_net(M.critic, xr, t, clip_emb, mot_win, capture_disc=True)
        d_gan = gan_d_loss_rel(logit_real, logit_fake) if relativistic else gan_d_loss(logit_real, logit_fake)
        loss = loss + gan_d_weight * d_gan
        log.update(gan_d=d_gan.detach().item(),
                   d_real=logit_real.detach().float().mean().item(),
                   d_fake=logit_fake.detach().float().mean().item())
        # 4) R1/R2 有限差分正则（paper Eq.5，λ=r1r2_weight, σ=r1r2_sigma）：扰动 real/fake、罚 logit 变化。
        #    复用已算的 logit_real/logit_fake（同 t/同窗），只额外 2 次带梯度 critic 前向（扰动样本）。
        if r1r2_weight > 0:
            xr_p = (xr + r1r2_sigma * torch.randn_like(xr)).to(M.dt)
            xt_p = (x_t + r1r2_sigma * torch.randn_like(x_t)).to(M.dt)
            _, _, logit_real_p = M.forward_net(M.critic, xr_p, t, clip_emb, mot_win, capture_disc=True)
            _, _, logit_fake_p = M.forward_net(M.critic, xt_p, t, clip_emb, mot_win, capture_disc=True)
            reg = gan_reg_r1r2(logit_real, logit_real_p, logit_fake, logit_fake_p)
            loss = loss + r1r2_weight * reg
            log["gan_reg"] = reg.detach().item()
    return loss, log

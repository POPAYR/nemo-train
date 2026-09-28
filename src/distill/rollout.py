"""
Phase 3: self-forcing rollout（DECODER_DISTILL_PLAN.md §Phase 3，仿 Self-Forcing inference_with_trajectory）
=========================================================================================================
逐 block 跑 4 步采样（predict x0 → re-noise 到下一步），但：
  - 梯度只在「**随机连续 grad_window 帧窗口 [k,k+W)**」内回传（其余 block no_grad）：
    · teacher 只在 ≤24 帧上训练（24 帧滑窗，PE max 32）→ DMD 打分必须限制在 ≤24 帧窗，
      否则喂 teacher 64 帧是 OOD、score 不准（2026-07-01 修正）；
    · **任意起点**的随机窗可跨 3 个 block 边界，随训练覆盖所有边界；
    · 只 grad 窗口重叠的少数 block → 显存也降。
  - block 去噪干净后用 commit 前向（context_noise，no_grad）写干净 KV cache（给后续 block 当历史）。
generator = M.generator（因果，TemporalCausalControl stream）。
返回 x0 rollout（**全长 L**，student 仍练长流式）+ grad 窗口 (k, W)（无梯度时 None）。
前置: 调用前需 M.set_reference(...) 设好 reference bank。
"""
import torch


def self_forcing_rollout(M, noise, clip_emb, motion, denoising_step_list,
                         block_size=8, grad_window=None, context_noise=0, full_steps=False):
    """
    noise:  [B, C, F, H, W]  纯噪声（= 首步 timestep 的 x）
    motion: [B, F, 32, 16]   逐帧 motion 条件
    grad_window: int W → 随机取连续 [k,k+W) 帧保留梯度（返回 (k,W)）；None/0 → 全 no_grad（critic/推理）。
    返回:   x0_out [B,C,F,H,W]（梯度只在随机窗内）, win=(k,W) 或 None
    """
    B, C, F_, H, W = noise.shape
    dev = noise.device
    sched = M.scheduler
    nstep = len(denoising_step_list)
    assert F_ % block_size == 0, (F_, block_size)
    num_blocks = F_ // block_size

    # 随机连续窗（任意起点，可跨 3 个边界）；无 window → 无梯度（推理/critic）。
    # 注：full_steps + grad_window 用于「可微渲染」(DMD-through-G)：每 block 跑满步 + 窗内可微。
    if not grad_window:
        win_k, Wn = None, 0
    else:
        Wn = int(grad_window)
        assert Wn <= F_, (Wn, F_)
        win_k = int(torch.randint(0, F_ - Wn + 1, (1,)).item())

    def _grad_block(cur):
        # 该 block [cur,cur+block_size) 是否与梯度窗 [win_k,win_k+Wn) 相交
        if win_k is None:
            return False
        return (cur < win_k + Wn) and (cur + block_size > win_k)

    # 训练：每 block 随机一个 exit step；推理 full_steps：每 block 跑满全部步（最后一步退出）
    if full_steps:
        exit_flags = [nstep - 1] * num_blocks
    else:
        exit_flags = torch.randint(0, nstep, (num_blocks,)).tolist()

    ctrl = M.gen_causal
    ctrl.set_mode("stream"); ctrl.reset_cache()

    outs = []
    cur = 0
    for blk in range(num_blocks):
        sl = slice(cur, cur + block_size)
        noisy = noise[:, :, sl]
        mot_b = motion[:, sl]
        grad_block = _grad_block(cur)
        ctrl.set_offset(cur)

        x0 = None
        for i, t_i in enumerate(denoising_step_list):
            t = torch.full((B,), int(t_i), device=dev, dtype=torch.long)
            ctrl.set_commit(False)                      # 去噪中间步只读缓存、不污染
            exit_i = (i == exit_flags[blk])
            if exit_i and grad_block:
                _, x0 = M.forward_net(M.generator, noisy, t, clip_emb, mot_b)
            else:
                with torch.no_grad():
                    _, x0 = M.forward_net(M.generator, noisy, t, clip_emb, mot_b)
            if exit_i:
                break
            # predict x0 → re-noise 到下一（更低）timestep
            with torch.no_grad():
                t_next = torch.full((B,), int(denoising_step_list[i + 1]), device=dev, dtype=torch.long)
                noisy = sched.add_noise(x0, torch.randn_like(x0), t_next).to(noise.dtype)

        outs.append(x0)

        # commit：用干净 x0 在 context_noise 跑一次写干净 KV（给后续 block 当历史）
        ctrl.set_commit(True)
        with torch.no_grad():
            t_ctx = torch.full((B,), int(context_noise), device=dev, dtype=torch.long)
            noisy_ctx = x0.detach()
            if context_noise > 0:
                noisy_ctx = sched.add_noise(noisy_ctx, torch.randn_like(noisy_ctx), t_ctx).to(noise.dtype)
            M.forward_net(M.generator, noisy_ctx, t_ctx, clip_emb, mot_b)
        cur += block_size

    ctrl.set_commit(True); ctrl.set_mode("off")        # 复位
    x0_out = torch.cat(outs, dim=2)                     # [B,C,F,H,W]
    win = (win_k, Wn) if win_k is not None else None
    return x0_out, win

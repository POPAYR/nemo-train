"""
flow-space self-forcing rollout(rectified-flow 版 rollout.py)
==============================================================
与 rollout.self_forcing_rollout 同构,只把 DDPM ε 换成 flow v:
  - 每 block 逐步去噪:v,x0 = G(z_σ, σ);到 exit 步取 x0(带梯度)。
  - 步间 re-noise 用 flow 前向:z ← (1-σ_next)·x0 + σ_next·ε(新鲜 ε,self-forcing/diffusion-forcing 风格)。
  - block 去噪干净后用 context_sigma(≈0)commit 一次,写干净 KV cache 给后续 block 当历史。
sigma_list: 少步 σ 网格(高→低,如 N=4=[1.0,0.667,0.334,0.001]),末步 exit 出干净 x0。
梯度只在随机连续 grad_window 帧窗内回传(teacher/critic 打分 ≤24 帧,分布内)。
前置:调用前需 M.set_reference(...) 设好 reference bank;M.objective=="flow"。
"""
import torch
from .flow_math import flow_add_noise


def flow_rollout(M, noise, clip_emb, motion, sigma_list,
                 block_size=8, grad_window=None, context_sigma=0.0, full_steps=False):
    """
    noise:  [B,C,F,H,W] 纯噪声(σ=1 的 z)。
    motion: [B,F,32,16] 逐帧 motion 条件。
    sigma_list: list[float] 少步 σ(高→低),len=N。
    grad_window: int W → 随机连续 [k,k+W) 帧保留梯度(返回 (k,W));None/0 → 全 no_grad(critic/推理)。
    返回: x0_out [B,C,F,H,W](梯度只在窗内), win=(k,W) 或 None
    """
    B, C, F_, H, W = noise.shape
    dev = noise.device
    nstep = len(sigma_list)
    assert F_ % block_size == 0, (F_, block_size)
    num_blocks = F_ // block_size

    if not grad_window:
        win_k, Wn = None, 0
    else:
        Wn = int(grad_window)
        assert Wn <= F_, (Wn, F_)
        win_k = int(torch.randint(0, F_ - Wn + 1, (1,)).item())

    def _grad_block(cur):
        if win_k is None:
            return False
        return (cur < win_k + Wn) and (cur + block_size > win_k)

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
        z = noise[:, :, sl]                          # σ=1 的纯噪声块
        mot_b = motion[:, sl]
        grad_block = _grad_block(cur)
        ctrl.set_offset(cur)

        x0 = None
        for i, sig in enumerate(sigma_list):
            ctrl.set_commit(False)                   # 去噪中间步只读缓存、不污染
            exit_i = (i == exit_flags[blk])
            if exit_i and grad_block:
                _, x0 = M.forward_net_flow(M.generator, z, sig, clip_emb, mot_b)
            else:
                with torch.no_grad():
                    _, x0 = M.forward_net_flow(M.generator, z, sig, clip_emb, mot_b)
            if exit_i:
                break
            # predict x0 → flow re-noise 到下一个更低 σ(新鲜噪声)
            with torch.no_grad():
                sig_next = sigma_list[i + 1]
                z = flow_add_noise(x0, torch.randn_like(x0), sig_next).to(noise.dtype)

        outs.append(x0)

        # commit:用干净 x0 在 context_sigma(≈0)跑一次写干净 KV
        ctrl.set_commit(True)
        with torch.no_grad():
            z_ctx = x0.detach()
            if context_sigma > 0:
                z_ctx = flow_add_noise(z_ctx, torch.randn_like(z_ctx), context_sigma).to(noise.dtype)
            M.forward_net_flow(M.generator, z_ctx, context_sigma, clip_emb, mot_b)
        cur += block_size

    ctrl.set_commit(True); ctrl.set_mode("off")
    x0_out = torch.cat(outs, dim=2)
    win = (win_k, Wn) if win_k is not None else None
    return x0_out, win

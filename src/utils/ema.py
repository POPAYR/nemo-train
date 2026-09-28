"""训练权重的指数滑动平均(EMA)。

★ 为什么补这个(2026-09-17 发现):整条 flow 线(stage1/stage2/teacher-ft/ode_init)
  从来没有 EMA,只有 DMD 与 AR motion 那两条线有。后果是在 eff_bsz=4 的高梯度噪声下
  **权重自己在游走**:过拟合诊断实验里,固定 seed、固定验证 clip 的条件下,
  边界带闪烁在 step1000/1500 之间从 0.431 跳到 0.485(±7%),PSNR 从 31.31 退到 30.84。
  测量是确定性的,所以那不是测量噪声,是权重抖动 —— 它让任何单点判决都不可靠
  (当天因此判早三次,见 DATA.md §18)。

★ 影子放 GPU 还是 CPU:实测 330M 参数/546 张量,
  CPU fp32 影子 0.246 秒/次(占 9.07s/step 的 2.7%),
  GPU fp32 影子 0.014 秒/次(0.15%),代价是约 1.3~1.8GB 显存。
  默认放 GPU;显存吃紧时用 device="cpu" 退回。

★ 预热:直接用 decay=0.999 起步会让前 ~1000 步的 EMA 被随机初值拖住。
  用 min(decay, (1+step)/(10+step)) 让早期跟得紧、后期逐渐平滑(diffusers 同款做法)。
"""
from __future__ import annotations

import torch


class EMA:
    """只跟踪 requires_grad=True 的参数(我们各阶段都只训一部分网络)。"""

    def __init__(self, model, decay: float = 0.999, device: str = "auto", warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self._backup = None
        dev = None if device == "auto" else torch.device(device)
        self.shadow = {
            n: p.detach().float().clone().to(dev if dev is not None else p.device)
            for n, p in model.named_parameters() if p.requires_grad
        }

    def _cur_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, model):
        d = self._cur_decay()
        self.step += 1
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[n]
            s.mul_(d).add_(p.detach().float().to(s.device), alpha=1 - d)

    # ---- 验证时临时换上 EMA 权重 ----
    @torch.no_grad()
    def apply_to(self, model):
        """把 EMA 权重写进模型,原权重备份在内部。必须与 restore 成对使用。"""
        assert self._backup is None, "apply_to 已生效,重复调用会丢失原权重"
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n].to(p.device, p.dtype))

    @torch.no_grad()
    def restore(self, model):
        assert self._backup is not None, "restore 前必须先 apply_to"
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    # ---- 存取 ----
    def state_dict(self, dtype=torch.bfloat16):
        return {n: v.to(dtype).cpu() for n, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for n, v in sd.items():
            if n in self.shadow:
                self.shadow[n].copy_(v.float().to(self.shadow[n].device))


class _EMAScope:
    def __init__(self, ema, model):
        self.ema, self.model = ema, model

    def __enter__(self):
        if self.ema is not None:
            self.ema.apply_to(self.model)
        return self.model

    def __exit__(self, *a):
        if self.ema is not None:
            self.ema.restore(self.model)
        return False


def ema_weights(ema, model):
    """with ema_weights(ema, model): ...  —— 验证/采样期间临时用 EMA 权重,
    异常时也保证 restore(否则一次 val 崩溃会让训练继续用 EMA 权重跑下去)。"""
    return _EMAScope(ema, model)

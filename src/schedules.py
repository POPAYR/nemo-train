"""
schedules.py
============
训练课程与权重调度：
  - linear_warmup: 通用线性 warmup/ramp。
  - WeightSchedule: λ_video / λ_reg 等 loss 权重随 step 变化。
  - SelfForcingSchedule: scheduled-sampling 概率 p（teacher-forcing → self-forcing 渐进）。
  - SamplingStepSchedule: 可微采样步数随 step 变化（单步 warmup → 2-4 步精调）。

设计依据（见讨论）：
  * λ_video 从 0 ramp，先用 regression 锚定稳住地形，再逐步引入 video loss；
  * p 从 0 慢慢升到 p_max(<1)，先 teacher-forcing 热启动，再渐进 self-forcing 修 exposure bias；
  * 采样步数先 1（稳、便宜）再升到 2-4（训练分布=推理分布，保高频口型）。
"""

from dataclasses import dataclass


def linear_warmup(step: int, start_step: int, end_step: int,
                  start_val: float, end_val: float) -> float:
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    r = (step - start_step) / max(1, (end_step - start_step))
    return start_val + r * (end_val - start_val)


@dataclass
class WeightSchedule:
    """分段线性：[0,warmup_start] 取 start_val；[warmup_start,warmup_end] 线性到 end_val；之后 end_val。"""
    start_val: float
    end_val: float
    warmup_start: int = 0
    warmup_end: int = 0

    def __call__(self, step: int) -> float:
        if self.warmup_end <= self.warmup_start:
            return self.end_val
        return linear_warmup(step, self.warmup_start, self.warmup_end,
                             self.start_val, self.end_val)


@dataclass
class SelfForcingSchedule:
    """scheduled-sampling 概率 p：用模型自身输出替换 GT 历史的概率。
    p: 0 → p_max，在 [start_step, end_step] 线性 ramp。enabled=False 时恒为 0（纯 teacher-forcing）。"""
    enabled: bool = True
    p_max: float = 0.5
    start_step: int = 2000
    end_step: int = 12000

    def __call__(self, step: int) -> float:
        if not self.enabled:
            return 0.0
        return linear_warmup(step, self.start_step, self.end_step, 0.0, self.p_max)


@dataclass
class SamplingStepSchedule:
    """可微采样步数：阶段性整数调度。step < switch_step 用 start_steps，之后用 end_steps。"""
    start_steps: int = 1
    end_steps: int = 4
    switch_step: int = 8000

    def __call__(self, step: int) -> int:
        return self.start_steps if step < self.switch_step else self.end_steps


def build_schedules(cfg):
    """从 OmegaConf 的 loss/curriculum 段构造调度器。"""
    lv = cfg.loss.lambda_video
    lr = cfg.loss.lambda_reg
    lvel = cfg.loss.get("lambda_vel", {"start_val": 0.0, "end_val": 0.0})
    sf = cfg.curriculum.self_forcing
    ss = cfg.curriculum.sampling_steps
    la = cfg.loss.get("lambda_anchor", {"start_val": 1.0, "end_val": 0.5,
                                    "warmup_start": 0, "warmup_end": 0})
    return {
        "lambda_video": WeightSchedule(lv.start_val, lv.end_val, lv.warmup_start, lv.warmup_end),
        "lambda_reg":   WeightSchedule(lr.start_val, lr.end_val, lr.warmup_start, lr.warmup_end),
        "lambda_vel":   WeightSchedule(lvel["start_val"], lvel["end_val"],
                                    lvel.get("warmup_start", 0), lvel.get("warmup_end", 0)),
        "lambda_anchor": WeightSchedule(la["start_val"], la["end_val"],
                                        la.get("warmup_start", 0), la.get("warmup_end", 0)),
        "self_forcing": SelfForcingSchedule(sf.enabled, sf.p_max, sf.start_step, sf.end_step),
        "sampling_steps": SamplingStepSchedule(ss.start_steps, ss.end_steps, ss.switch_step),
    }
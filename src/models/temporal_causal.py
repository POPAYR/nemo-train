# *************************************************************************
# Phase 1 (DECODER_DISTILL_PLAN.md): 把 X-Nemo 的双向 temporal_modules 改造成
# block-causal + KV-cache 流式。只作用于 Temporal_Self 的 VersatileAttention；
# motion_modules(Spatial_Cross) 与 spatial UNet 不受影响。
#
# 用法（仿 ReferenceAttentionControl 的模块级状态注入，无需改 UNet forward 签名）:
#   ctrl = TemporalCausalControl(denoising_unet, block_size=8, window=0)
#   # 训练态（整窗 + block-causal mask）:
#   ctrl.set_mode("train");  out = unet(x_full, ...)
#   # 流式态（逐 block KV-cache）:
#   ctrl.set_mode("stream"); ctrl.reset_cache()
#   for s in range(0, F, K):
#       ctrl.set_offset(s);  out_blk = unet(x_full[:, :, s:s+K], ...)
#   # 关闭（回到原始双向，teacher 用）:
#   ctrl.disable()
# *************************************************************************
from .motion_module import VersatileAttention


def collect_temporal_self_attns(unet):
    """找出 UNet 里所有「时序自注意力」VersatileAttention（即破坏因果性的模块）。"""
    mods = []
    for m in unet.modules():
        if isinstance(m, VersatileAttention) \
                and m.attention_mode == "Temporal" and not m.is_cross_attention:
            mods.append(m)
    return mods


class TemporalCausalControl:
    def __init__(self, unet, block_size: int, window: int = 0):
        """
        block_size: 一个因果 block 含几个 latent 帧（= num_frame_per_block，部署用 8）。
        window:     局部注意力窗口（帧）。0=全历史（仅适合 ≤PE max_len=32 的序列）；
                    >0=只看最近 window 帧（流式有界显存，须 ≤32 以匹配 PE）。
        """
        self.attns = collect_temporal_self_attns(unet)
        self.block_size = block_size
        self.window = window
        for m in self.attns:
            m._causal_mode = "off"
            m._causal_block_size = block_size
            m._causal_window = window
            m._frame_offset = 0
            m._kv_cache = None

    def __len__(self):
        return len(self.attns)

    def set_mode(self, mode: str):
        # bidir = 全双向但走同一条注意力实现(为了同样使用 RoPE);off = 走原 diffusers processor
        assert mode in ("off", "train", "stream", "bidir")
        for m in self.attns:
            m._causal_mode = mode

    def set_rope(self, enabled: bool = True):
        """位置编码切换:True=RoPE(旋转 q/k,相对、可外推、train==stream 严格一致)
        False=原加性正弦绝对 PE。改这个等于换架构,需重新微调 temporal 模块。"""
        for m in self.attns:
            m._rope = bool(enabled)

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        for m in self.attns:
            m._causal_block_size = block_size

    def set_window(self, window: int):
        self.window = window
        for m in self.attns:
            m._causal_window = window

    def set_offset(self, offset: int):
        """流式逐 block：设置当前 block 的绝对起始帧（用于 PE 对齐）。"""
        for m in self.attns:
            m._frame_offset = offset

    def set_commit(self, commit: bool):
        """流式：是否把当前帧 K/V 写入缓存。多步去噪的中间步设 False（只读），
        block 去噪干净后用一次 commit=True 的前向写入干净 K/V。1 步生成保持 True。"""
        for m in self.attns:
            m._causal_commit = commit

    def reset_cache(self):
        for m in self.attns:
            m._kv_cache = None
        self.set_offset(0)

    def disable(self):
        self.set_mode("off")
        self.reset_cache()


def set_temporal_rope(unet, enabled: bool = True, mode: str = "bidir"):
    """给不建 TemporalCausalControl 的场景(双向 teacher 微调/渲染)用的独立开关。
    mode="bidir":全双向 + RoPE(替代原 processor 路径的加性绝对 PE)。"""
    attns = collect_temporal_self_attns(unet)
    for m in attns:
        m._rope = bool(enabled)
        m._causal_mode = mode
        m._frame_offset = 0
        m._causal_window = 0
        m._kv_cache = None
    return len(attns)

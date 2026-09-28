# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under AnimateDiff, with the full license text
# available at https://github.com/guoyww/AnimateDiff/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.
# *************************************************************************
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn


# --------------------------- RoPE(时序维,1-D)---------------------------
# 替代原「加性正弦绝对 PE」。动机:绝对 PE 把质量绑死在训练长度上(实测 ε/flow teacher 离开各自
# 训练长度后锐度与时序一致性都显著退化),且流式推理时 offset 无限增长 → 永远在未训练位置。
# RoPE 是相对的:q_i·k_j 只依赖 (i-j),所以
#   ① 训练(整段 0..L-1 + block-causal mask)与推理(逐 block offset..offset+B-1 + KV cache)
#      给出**完全相同**的注意力分数 —— train-infer 零错配;
#   ② 配合局部窗口,相对距离恒有界 → 任意长度流式。
# 约定与官方 Self-Forcing/Wan 一致(wan/modules/model.py::rope_apply):
#   - 施加在 **q 和 k**、投影之后、逐 head;
#   - 索引用**绝对帧位置**(推理时 = current_start_frame + i,官方 causal_rope_apply 同);
#   - 相邻维两两配对做复数旋转(view_as_complex 的交错约定)。
_ROPE_CACHE = {}


def _rope_cos_sin(head_dim: int, max_pos: int, device, theta: float = 10000.0):
    key = (head_dim, str(device), theta)
    ent = _ROPE_CACHE.get(key)
    if ent is None or ent[0].shape[0] < max_pos:
        need = max(max_pos, 256)
        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64)
        inv = 1.0 / (theta ** (idx / head_dim))                 # [dh/2]
        pos = torch.arange(need, device=device, dtype=torch.float64)
        ang = torch.outer(pos, inv)                             # [need, dh/2]
        ent = (ang.cos(), ang.sin())
        _ROPE_CACHE[key] = ent
    return ent


def apply_rope_1d(x, pos, theta: float = 10000.0):
    """x: [B, heads, L, dh];pos: [L] long(绝对帧位置)。返回同形状旋转后张量。
    ★ 旋转在 fp32 内计算再转回原 dtype:bf16 只有 8 位尾数(相对精度~0.4%),
      而 RoPE 是逐元素乘加、误差直接进注意力分数。官方 rope_apply 用 float64 复数乘,同理。"""
    dh = x.shape[-1]
    assert dh % 2 == 0, f"RoPE 需要偶数 head_dim,得到 {dh}"
    cos, sin = _rope_cos_sin(dh, int(pos.max().item()) + 1, x.device, theta)
    c = cos[pos].float().view(1, 1, x.shape[2], dh // 2)
    s = sin[pos].float().view(1, 1, x.shape[2], dh // 2)
    xf = x.float()
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    return torch.stack([xe * c - xo * s, xe * s + xo * c], dim=-1).flatten(-2).to(x.dtype)


def build_block_causal_mask(f: int, block_size: int, device, window: int = 0):
    """布尔注意力掩码 [f, f]，True=允许 attend。
    块内双向、跨块只看过去：frame i 可 attend frame j  iff  block(j) <= block(i)。
    window>0 时再叠加局部窗口：仅允许 i-window < j <= i 范围（块对齐），用于流式有界显存。
    """
    idx = torch.arange(f, device=device)
    blk = idx // block_size
    mask = blk.unsqueeze(1) >= blk.unsqueeze(0)  # [i,j] True if block(i)>=block(j)
    if window and window > 0:
        # 只看最近 window 帧（含自身块）：j 的块 > i 的块-floor(window/block) 时才允许
        past_blocks = max(1, window // block_size)
        mask = mask & (blk.unsqueeze(0) > (blk.unsqueeze(1) - past_blocks))
    return mask


def zero_module(module):
    # Zero out the parameters of a module and return it.
    assert isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear), type(module)
    for p in module.parameters():
        p.detach().zero_()
    return module

def random_module(m):
    assert isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear), type(m)
    # Initialize weights with He initialization and zero out the biases
    n = (m.kernel_size[0] * m.kernel_size[1] * m.in_channels) if isinstance(m, nn.Conv2d) else m.in_features
    nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2. / n))
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(in_channels, motion_module_type: str, motion_module_kwargs: dict):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            **motion_module_kwargs,
        )
    elif motion_module_type == "RefImage_Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            skip_ref_image=True,
            **motion_module_kwargs,
        )
    elif motion_module_type == "RefImageCond_Vanilla":
        return VanillaTemporalModule(
            in_channels=in_channels,
            cond_ref_image=True,
            **motion_module_kwargs,
        )
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):

    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        num_transformer_block=2,
        attention_block_types=("Temporal_Self", "Temporal_Self"),
        cross_attention_dim=768,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1,
        zero_initialize=True,
        skip_ref_image=False,
        cond_ref_image=False,
    ):
        super().__init__()
        self.skip_ref_image = skip_ref_image
        self.cond_ref_image = cond_ref_image

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels
            // num_attention_heads
            // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_attention_dim=cross_attention_dim,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(
                self.temporal_transformer.proj_out
            )

    def set_use_cross_frame_attention(self, value):
        self.skip_ref_image = value

    def forward(
        self,
        input_tensor,
        temb,
        encoder_hidden_states,
        attention_mask=None,
        anchor_frame_idx=None,
        debug=False
    ):
        hidden_states = input_tensor
        if self.skip_ref_image:
            # if input_tensor.shape[2] > 1:
                hidden_states, ref_hidden_states = input_tensor[:, :, :-1], input_tensor[:, :, -1:]

        hidden_states = self.temporal_transformer(
            hidden_states, encoder_hidden_states, attention_mask, debug=debug
        )

        output = hidden_states
        if self.skip_ref_image:
            # if input_tensor.shape[2] > 1:
                output = torch.cat([output, ref_hidden_states], dim=2)
        elif self.cond_ref_image:
            output = torch.cat([output[:, :, :-1], input_tensor[:, :, -1:]], dim=2)
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, debug=False):
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        if encoder_hidden_states is not None and encoder_hidden_states.ndim == 4:
            assert encoder_hidden_states.shape[1] == video_length, (video_length, encoder_hidden_states.shape)
            encoder_hidden_states = rearrange(encoder_hidden_states, "b d n c -> (b d) n c",)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, height * weight, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
            )

        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if False:
            print(
                'TemporalModule',
                hidden_states.shape,
                # round(torch.abs(residual).mean().item(), 6),
                # round(torch.abs(residual).max().item(), 6),
                # round(torch.abs(hidden_states).mean().item(), 6),
                # round(torch.abs(hidden_states).max().item(), 6),
            )
            # hidden_states *= 0
        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output


class TemporalTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        proj_out_dim=None,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim
                    if block_name.endswith("_Cross")
                    else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

        self.proj_out = nn.Linear(dim, proj_out_dim) if proj_out_dim is not None else None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        att_flag=False
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            if att_flag:
                print(
                    'block',
                    round(torch.abs(hidden_states).mean().item(), 6),
                    round(torch.abs(norm_hidden_states).mean().item(), 6),
                )
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if attention_block.is_cross_attention
                    else None,
                    video_length=video_length,
                    att_flag=att_flag
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states if self.proj_out is None else self.proj_out(hidden_states)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        # persistent=False: 正弦 PE 是确定性公式, 不存/不从 ckpt 加载 → 可自由扩展长度,
        # 且位置 0..max_len-1 与原始逐元素一致 (扩展不破坏 ≤32 帧的对拍)。
        self.register_buffer("pe", self._make_pe(max_len), persistent=False)

    def _make_pe(self, length):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, length, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, offset: int = 0):
        # offset: 绝对帧起点。bidirectional/train 用 0；流式逐 block 时用该 block 的绝对起始帧，
        # 使缓存帧保留正确的绝对位置编码（这是 train==stream 数值一致的前提）。
        need = offset + x.size(1)
        if need > self.pe.size(1):
            # 流式 >32 帧时自动扩展正弦 PE（位置 0..31 不变, 仅追加 OOD 位置, 由蒸馏适配）。
            self.pe = self._make_pe(need).to(device=self.pe.device, dtype=self.pe.dtype)
        x = x + self.pe[:, offset: offset + x.size(1)]
        return self.dropout(x)


class VersatileAttention(Attention):
    def __init__(
        self,
        attention_mode=None,
        cross_frame_attention_mode=None,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode in ["Temporal", "Spatial"], attention_mode

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None

        self.pos_encoder = (
            PositionalEncoding(
                kwargs["query_dim"],
                dropout=0.0,
                max_len=temporal_position_encoding_max_len,
            )
            if (temporal_position_encoding and attention_mode == "Temporal")
            else None
        )

    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Optional[Callable] = None,
    ):
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            # XFormersAttnProcessor corrupts video generation and work with Pytorch 1.13.
            # Pytorch 2.0.1 AttnProcessor works the same as XFormersAttnProcessor in Pytorch 1.13.
            # You don't need XFormersAttnProcessor here.
            # processor = XFormersAttnProcessor(
            #     attention_op=attention_op,
            # )
            processor = AttnProcessor()
        else:
            processor = AttnProcessor()

        self.set_processor(processor)

    def _causal_temporal_attn(self, hidden_states, video_length, mode):
        """因果时序自注意力（替代双向 self-attn），直接用 self.to_q/k/v/out + SDPA。
        hidden_states: [(b d), f, c]，f = 本次前向的帧数。
        - mode="train"：整窗 f 帧 + block-causal mask（块内双向、跨块只看过去）。
        - mode="stream"：当前 block 的 f 帧做 query，K/V = [KV-cache(历史) + 当前]，无 mask。
        - mode="bidir" ：整窗 f 帧全双向（双向 teacher；走本函数以便同样使用 RoPE）。
        二者在相同 (block_size, window, 位置编码) 下逐帧数值一致——这是 train==推理的正确性不变量。
        位置编码二选一：_rope=True 用 RoPE（旋转 q/k，相对、可外推）；否则用原加性正弦绝对 PE。
        """
        h = hidden_states
        f = h.shape[1]
        offset = getattr(self, "_frame_offset", 0)
        block_size = getattr(self, "_causal_block_size", None) or f
        window = getattr(self, "_causal_window", 0)
        use_rope = getattr(self, "_rope", False)

        if self.pos_encoder is not None and not use_rope:
            h = self.pos_encoder(h, offset=offset)  # 绝对位置编码（流式按 offset 对齐）

        q = self.to_q(h); k = self.to_k(h); v = self.to_v(h)
        B = h.shape[0]; heads = self.heads; dh = q.shape[-1] // heads

        def split(x):  # [B, L, inner] -> [B, heads, L, dh]
            return x.view(B, x.shape[1], heads, dh).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        if use_rope:
            # 绝对帧位置 offset..offset+f-1（与官方 causal_rope_apply(start_frame=...) 同）。
            # ★ 只旋转当前这批 q/k；缓存里的 k 早已在写入时按其**当时的绝对位置**旋转过，
            #   因此拼接后 q_i·k_j 自动只依赖 (i-j) —— 这是 train/stream 数值一致的关键。
            pos = torch.arange(offset, offset + f, device=h.device)
            q = apply_rope_1d(q, pos)
            k = apply_rope_1d(k, pos)

        if mode == "bidir":
            attn_mask = None
        elif mode == "stream":
            cache = getattr(self, "_kv_cache", None)
            commit = getattr(self, "_causal_commit", True)
            if cache is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            if window and window > 0 and k.shape[2] > window:
                k = k[:, :, -window:]; v = v[:, :, -window:]
            if commit:
                # commit=True: 把当前(干净)K/V 写入缓存(默认/1步生成即提交)。
                # commit=False: 多步去噪中间步只读 [缓存+当前], 不污染缓存(等 block 去噪干净后再单独 commit)。
                self._kv_cache = {"k": k, "v": v}
            attn_mask = None
        else:  # "train"
            attn_mask = build_block_causal_mask(f, block_size, h.device, window=window)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, f, heads * dh)
        out = self.to_out[0](out)   # Linear
        out = self.to_out[1](out)   # Dropout (eval=identity)
        return out

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        bank=None,
        att_flag=False,
        **cross_attention_kwargs,
    ):
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]  # d means HxW
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

            if encoder_hidden_states is not None:
                if not encoder_hidden_states.shape[0] == hidden_states.shape[0]:
                    encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)

        causal_mode = getattr(self, "_causal_mode", "off")
        if (causal_mode != "off" and self.attention_mode == "Temporal"
                and not self.is_cross_attention):
            # 因果时序注意力：train=block-causal mask 整窗前向；stream=KV-cache 逐 block。
            hidden_states = self._causal_temporal_attn(hidden_states, video_length, causal_mode)

        elif bank is not None and self.attention_mode == "Temporal" and not self.is_cross_attention:
            # motion_frames作为之前的帧，引入motion module进行condition
            modify_norm_hidden_states = torch.cat(bank + [hidden_states], dim=1)

            if self.pos_encoder is not None:
                modify_norm_hidden_states = self.pos_encoder(modify_norm_hidden_states)

            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=modify_norm_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )  # 改为cross-att

        else:
            if self.pos_encoder is not None:
                hidden_states = self.pos_encoder(hidden_states)
            inp = hidden_states
            hidden_states = self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if att_flag:
                print(
                    'ver_att',
                    round(torch.abs(inp).mean().item(), 6),
                    round(torch.abs(encoder_hidden_states).mean().item(), 6),
                    round(torch.abs(hidden_states).mean().item(), 6),
                )

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

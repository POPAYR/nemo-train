"""
双流流式推理 Pipeline（d8 实时交付）
=====================================================================================
两条独立的因果流，用一个有界队列串起来，视频流只滞后 motion 流 **一个 block** 的延迟：

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  音频/文本 ── wav2vec2 / umt5 ──►  条件 (text_emb, audio_emb, local_audio)     │
  └─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  ┌──────────────────── MOTION 流（生产者，逐帧）────────────────────┐
           │  MotionTransformer.generate 的流式版：每帧一次 KV-cache 前向 +    │
           │  diffloss 采样 → 反归一化 motion latent[512] → push 进队列        │
           └──────────────────────────────────────────────────────────────────┘
        │        每攒够 block_size(=8) 帧
        ▼  ┌──────────────────── VIDEO 流（消费者，逐 block）─────────────────┐
           │  从队列取 8 帧 motion → reshape[1,8,32,16] → 因果解码器 4 步渲染  │
           │  （跨 block 保持干净 KV，offset 递增）→ x0 潜码[1,C,8,H,W]        │
           │  → SVD-VAE 解码 → 像素帧[8,3,H,W] → yield / 编码器               │
           └──────────────────────────────────────────────────────────────────┘

延迟分析（首帧）:
  first_pixel_latency ≈ block_size × t_motion_frame  +  (4×UNet3D block 渲染 + VAE 解码)
  而不是整段的  L × t_motion_frame  +  full_render 。即 motion 只需领先 8 帧，视频就开跑。

单 GPU 现实：两流在 GPU 上仍串行（CUDA 每 stream 顺序执行），双流的收益是
  ① 延迟——早出帧，不等整段 motion；
  ② 流水线——VAE 解码放独立 CUDA stream + 输出编码放独立线程，与下一 block 的
     motion/渲染 计算重叠，隐藏 CPU/编码开销。
  motion 帧极便宜（小 transformer+diffloss），瓶颈在视频 block 渲染，所以让 motion
  始终领先 1 block 几乎不增加总时长。

用法见文件末 `__main__` / `run_stream()`。

推理加速结论（A100 80GB, cc8.0, torch2.1.1+cu118, bf16 实测）:
  · fp16：Ampere 上 fp16 与 bf16 tensor-core 同吞吐 → 不换（bf16 已最优、无溢出风险）。
  · TF32/cudnn.benchmark/SDPA-flash：UNet3D 本就 bf16+FlashAttn-2 → 增益≈0（仍默认开，零风险）。
  · ★ 有界因果窗 causal_window=24（匹配 dmd2_win24 蒸馏）是真正的关键：
      window=0(全历史) → K/V 每 block 无界增长，耗时 1549→1790ms 持续爬升，且 >32 帧越界(PE)；
      window=24        → 每 block K/V 恒定 24 帧，耗时恒定 ~1600ms(=5.0fps)、显存有界、in-distribution。
  · torch.compile(generator)：见 build_pipeline compile_gen（1.7B UNet3D 计算受限，收益有限；实测见 bench_stream）。
  · 剩余实时杠杆在渲染本身：更少去噪步 / 蒸更小解码器（d8 交付时优化）。
"""
import threading
import queue
import time
import contextlib
import torch

from .models import DMD2Models


# ============================================================================
#  推理加速：全局数值/kernel 开关 + SDPA flash 内核上下文
# ----------------------------------------------------------------------------
#  环境实测(A100 80GB, cc8.0, torch2.1.1+cu118)：
#   · fp16 vs bf16 在 Ampere 上 tensor-core 同吞吐 → 不换 fp16（bf16 已最优、无溢出风险）
#   · SDPA 默认 AttnProcessor2_0 已走 FlashAttention-2；这里再显式 pin flash 内核
#   · TF32 / cudnn.benchmark / matmul 'high' 对 conv+matmul 零风险提速
# ============================================================================
def enable_fast_math():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True                                # 固定 shape（block=8）→ 选最优 conv 算法
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def sdpa_flash_ctx():
    """强制 SDPA 用 flash 内核（关掉 math 回退）。torch2.1 用 sdp_kernel 上下文。"""
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        return contextlib.nullcontext()


# ============================================================================
#  MOTION 流：把 MotionTransformer.generate 拆成「逐帧可迭代」的生产者
# ============================================================================
class MotionStream:
    """逐帧流式 motion 生成器（保留 KV-cache，等价于 generate() 但一次吐一帧）。

    条件 (text_emb/audio_emb/local_audio_feat) 由外部预处理好传入。
    输出：**反归一化**的 motion latent，每帧 [B, 512]（与训练里 m_hat 同空间）。
    """
    def __init__(self, motion_model, device, dtype=torch.bfloat16,
                 cfg_audio=2.0, cfg_text=1.5, cfg_schedule="constant",
                 use_lcm=True, lcm_steps=4, temperature=0.9, num_sampling_steps=None):
        self.M = motion_model.eval()
        self.device, self.dtype = device, dtype
        self.cfg_audio, self.cfg_text, self.cfg_schedule = cfg_audio, cfg_text, cfg_schedule
        self.use_lcm, self.lcm_steps, self.temperature = use_lcm, lcm_steps, temperature
        self.num_sampling_steps = num_sampling_steps

    @torch.no_grad()
    def stream(self, text_emb, audio_emb, local_audio_feat, seq_len, first_frame=None):
        """生成器：逐帧 yield 反归一化 motion latent [B, 512]。"""
        M = self.M
        device, dtype = self.device, self.dtype
        bsz = text_emb.shape[0]
        if self.num_sampling_steps is not None:
            old_steps = M.diffloss.num_sampling_steps
            M.diffloss.num_sampling_steps = int(self.num_sampling_steps)

        try:
            with torch.cuda.amp.autocast(dtype=dtype, enabled=(device.type == "cuda")):
                fusion_latents = M.fusion_net(text_emb, audio_emb)
                zero_fusion = torch.zeros_like(fusion_latents)
                local_audio_feat = M.audio_proj(local_audio_feat)         # [B, L_seq, L, D]
                use_cfg = (self.cfg_audio != 1.0) or (self.cfg_text != 1.0)

                # CFG 三分支固定条件（full / no_audio / uncond），与 generate() 一致
                if use_cfg:
                    fusion_in = torch.cat([fusion_latents, fusion_latents, zero_fusion], dim=0)
                else:
                    fusion_in = fusion_latents

                from model.armodel import SelfAttnKVCache, CrossAttnKVCache
                self_caches = [SelfAttnKVCache() for _ in M.layers]
                fusion_caches = [CrossAttnKVCache() for _ in M.layers]

                # 起始 token（frame 0）
                if first_frame is None:
                    cur = torch.zeros(bsz, 1, M.motion_dim, device=device, dtype=dtype)
                else:
                    ff = M.normalize(first_frame.to(device=device, dtype=dtype))
                    ff = ff.unsqueeze(1) if ff.dim() == 2 else ff
                    cur = ff[:, -1:].contiguous()

                for t in range(seq_len):
                    local_t = local_audio_feat[:, t:t + 1]
                    if use_cfg:
                        x_in = torch.cat([cur, cur, cur], dim=0)
                        zero_local = torch.zeros_like(local_t)
                        local_in = torch.cat([local_t, zero_local, zero_local], dim=0)
                    else:
                        x_in, local_in = cur, local_t

                    h = M.motion_proj(x_in)
                    for layer, sc, fc in zip(M.layers, self_caches, fusion_caches):
                        h = layer.forward_cached(h, fusion_in, local_in, sc, fc, M.max_len)
                    last_feat = M.norm(h[:, -1])

                    if self.cfg_schedule == "linear":
                        prog = (t + 1) / seq_len
                        ca = 1.0 + (self.cfg_audio - 1.0) * prog
                        ct = 1.0 + (self.cfg_text - 1.0) * prog
                    else:
                        ca, ct = self.cfg_audio, self.cfg_text

                    if self.use_lcm and self.lcm_steps and self.lcm_steps > 1:
                        sample = M.diffloss.sample_for_infer(
                            last_feat, bsz=bsz, num_steps=int(self.lcm_steps),
                            cfg_audio=ca, cfg_text=ct, use_cfg=use_cfg, temperature=self.temperature)
                    elif self.use_lcm:
                        sample = M.diffloss.sample_lcm_dual_cfg(
                            last_feat, bsz=bsz, cfg_audio=ca, cfg_text=ct,
                            use_cfg=use_cfg, temperature=self.temperature)
                    else:
                        sample = M.diffloss.sample_dual_cfg(
                            last_feat, bsz=bsz, cfg_audio=ca, cfg_text=ct, use_cfg=use_cfg)

                    cur = sample.unsqueeze(1)                            # 归一化空间，喂下一步
                    yield M.denormalize(cur[:, 0])                       # [B, 512] 反归一化，供渲染
        finally:
            if self.num_sampling_steps is not None:
                M.diffloss.num_sampling_steps = old_steps


# ============================================================================
#  VIDEO 流：逐 block 因果渲染器（跨 block 保持干净 KV，offset 递增）
# ============================================================================
class VideoStream:
    """把 self_forcing_rollout 的「逐 block 内核」拆出来做流式：
    每次喂一个 block 的 motion tokens，跨 block 保持因果 KV cache（stream 模式 + offset）。
    """
    def __init__(self, dmd_models: DMD2Models, vae, device, dtype=torch.bfloat16,
                 denoising_step_list=(999, 749, 499, 249), block_size=8,
                 context_noise=0, vae_scale=0.18215, motion_token_shape=(32, 16),
                 causal_window=0, reset_period=32, vae_stream=None):
        self.M = dmd_models
        self.vae = vae.eval()
        self.device, self.dtype = device, dtype
        self.dsl = list(denoising_step_list)
        self.block_size = block_size
        self.context_noise = context_noise
        self.vae_scale = vae_scale
        self.mts = tuple(motion_token_shape)
        # 因果窗（帧）：★ 解码器蒸馏时 TemporalCausalControl window=0（全历史，见 distill_decoder_dmd.py，
        #   "--window 24" 是 DMD 打分/梯度窗 grad_window，与因果注意力窗无关）。推理必须 =0，否则 rollout 发散成噪声。
        self.causal_window = int(causal_window)
        # ★ 分段重锚定：解码器只在 L=32 帧 rollout 上蒸馏（eval≤48），>~48 帧绝对 PE + KV 严重 OOD → 发散成噪声。
        #   且 PE 在 commit 时把绝对 offset 焊进缓存 K/V，一旦 offset 越界缓存本身被污染，滑窗救不了。
        #   唯一正确解：每 reset_period(≤32) 帧 reset(cache+offset=0)，同一 ref bank 重锚。块内保持时序连续，
        #   段边界可能微跳。=0 关闭（仅 ≤reset_period 短片用）。block 整数倍。
        self.reset_period = int(reset_period)
        self.clip_emb = None                                             # set_reference 时缓存
        self._lat_shape = None                                          # (C, H, W)
        self._offset = 0
        self.vae_stream = vae_stream                                     # 独立 CUDA stream 隐藏 VAE 解码

    @torch.no_grad()
    def set_reference(self, ref_latent, ref_img):
        """设身份 reference bank + 缓存 clip_emb + 记录潜码空间形状。ref_latent: [1,C,H,W]（未乘 scale）。"""
        ref_lat = ref_latent.to(self.device, self.dtype) * self.vae_scale
        self.clip_emb = self.M.clip_embed(ref_img.to(self.device, self.dtype))
        self.M.set_reference(ref_lat, self.clip_emb, 1)
        self._lat_shape = tuple(ref_lat.shape[1:])                       # (C, H, W)
        # 初始化因果解码器 stream 状态 + 有界因果窗（匹配 win24 蒸馏）
        self.M.gen_causal.set_mode("stream")
        self.M.gen_causal.set_window(self.causal_window)
        self.M.gen_causal.reset_cache()
        self._offset = 0

    @torch.no_grad()
    def render_block(self, motion_block):
        """motion_block: [block_size, 512] 反归一化 motion latent → 渲染 1 个 block。
        返回视频潜码 x0 [1, C, block_size, H, W]（已 /scale 前，未解码）。"""
        M, sched = self.M, self.M.scheduler
        bs = self.block_size
        C, H, W = self._lat_shape
        assert motion_block.shape[0] == bs, (motion_block.shape, bs)

        mot_tok = motion_block.reshape(1, bs, *self.mts).to(self.dtype)  # [1,8,32,16]
        noisy = torch.randn(1, C, bs, H, W, device=self.device, dtype=self.dtype)

        ctrl = M.gen_causal
        # ★ 分段重锚：offset 达到 reset_period 就清空缓存、offset 归零（每段都在训练 L=32 regime 内）
        if self.reset_period and self._offset >= self.reset_period:
            ctrl.reset_cache()
            self._offset = 0
        ctrl.set_offset(self._offset)

        x0 = None
        with sdpa_flash_ctx():                                           # UNet 注意力 pin flash 内核
            for i, t_i in enumerate(self.dsl):
                t = torch.full((1,), int(t_i), device=self.device, dtype=torch.long)
                ctrl.set_commit(False)                                   # 去噪中间步：只读缓存
                _, x0 = M.forward_net(M.generator, noisy, t, self.clip_emb, mot_tok)
                if i == len(self.dsl) - 1:
                    break
                t_next = torch.full((1,), int(self.dsl[i + 1]), device=self.device, dtype=torch.long)
                noisy = sched.add_noise(x0, torch.randn_like(x0), t_next).to(self.dtype)

            # commit：干净 x0 写入 KV cache，给后续 block 当历史
            ctrl.set_commit(True)
            t_ctx = torch.full((1,), int(self.context_noise), device=self.device, dtype=torch.long)
            noisy_ctx = x0.detach()
            if self.context_noise > 0:
                noisy_ctx = sched.add_noise(noisy_ctx, torch.randn_like(noisy_ctx), t_ctx).to(self.dtype)
            M.forward_net(M.generator, noisy_ctx, t_ctx, self.clip_emb, mot_tok)

        self._offset += bs
        return x0                                                        # [1,C,bs,H,W]

    @torch.no_grad()
    def decode_pixels(self, x0):
        """潜码 → 像素。SVD-VAE 逐帧解码；可放独立 CUDA stream 与下一 block 计算重叠。"""
        z = x0[0].permute(1, 0, 2, 3).contiguous() / self.vae_scale     # [bs, C, H, W]
        ctx = torch.cuda.stream(self.vae_stream) if self.vae_stream is not None else _null_ctx()
        with ctx:
            try:
                img = self.vae.decode(z, num_frames=z.shape[0]).sample   # 时序解码器签名
            except TypeError:
                img = self.vae.decode(z).sample                          # 普通 2D VAE 签名
        return img.clamp(-1, 1)                                          # [bs, 3, H2, W2] ∈ [-1,1]


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ============================================================================
#  编排：生产者(motion)/消费者(video) 两线程 + 有界队列，视频滞后 1 block
# ============================================================================
class DualStreamPipeline:
    def __init__(self, motion_stream: MotionStream, video_stream: VideoStream,
                 queue_blocks=4):
        self.mstream = motion_stream
        self.vstream = video_stream
        self.bs = video_stream.block_size
        # 队列上限 = queue_blocks 个 block 的 motion 帧（背压：视频慢时挡住 motion 跑飞）
        self.mot_q = queue.Queue(maxsize=queue_blocks * self.bs)
        self._err = None

    def _producer(self, text_emb, audio_emb, local_audio_feat, seq_len, first_frame):
        try:
            for frame in self.mstream.stream(text_emb, audio_emb, local_audio_feat,
                                             seq_len, first_frame):
                self.mot_q.put(frame[0])                                 # [512]
        except Exception as e:                                          # noqa
            self._err = e
        finally:
            self.mot_q.put(None)                                        # 结束哨兵

    def run(self, text_emb, audio_emb, local_audio_feat, ref_latent, ref_img,
            seq_len, first_frame=None):
        """启动双流，逐 block yield 像素帧 [bs, 3, H, W]。视频流仅滞后 1 个 block。"""
        self.vstream.set_reference(ref_latent, ref_img)

        prod = threading.Thread(
            target=self._producer,
            args=(text_emb, audio_emb, local_audio_feat, seq_len, first_frame),
            daemon=True)
        prod.start()

        buf, done = [], False
        while not done:
            frame = self.mot_q.get()
            if frame is None:
                done = True
            else:
                buf.append(frame)
            # 攒够一个 block（或流结束时凑不满则丢弃尾巴，保持 block 对齐）
            while len(buf) >= self.bs:
                block = torch.stack(buf[:self.bs], dim=0)                # [bs, 512]
                del buf[:self.bs]
                x0 = self.vstream.render_block(block)
                yield self.vstream.decode_pixels(x0)

        prod.join()
        if self._err is not None:
            raise self._err


# ============================================================================
#  便捷入口：一次装齐两模型 + VAE，跑一段
# ============================================================================
def build_pipeline(motion_model, device, dtype=torch.bfloat16,
                   gen_ckpt=None, block_size=8,
                   denoising_step_list=(999, 749, 499, 249),
                   motion_cfg=None, overlap_vae=True,
                   fast_math=True, compile_vae=False, compile_gen=False):
    """装配双流。motion_model 为已加载权重的 MotionTransformer。

    加速开关（默认只开零风险项）：
      fast_math    : TF32 + cudnn.benchmark + matmul 'high'（零风险，默认开）
      compile_vae  : torch.compile VAE 解码（静态 shape，安全；首块编译慢）
      compile_gen  : torch.compile 生成器 UNet3D（有风险：reference-control 钩子 +
                     增长 KV 缓存易触发重编译；dynamic=True + try 兜底，靠实测决定）
    """
    from diffusers import AutoencoderKLTemporalDecoder
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")

    if fast_math and device.type == "cuda":
        enable_fast_math()

    # 渲染器 ckpt 的键是 "generator"（不是 models.py 内部默认的 "denoising_unet"），
    # 照训练 DMDThroughG 的方式：gen_ckpt=None 构造，再手动灌 sd["generator"]。
    dmd = DMD2Models(device, dt=dtype, gen_ckpt=None, block_size=block_size)
    if gen_ckpt is not None:
        sd = torch.load(gen_ckpt, map_location="cpu")
        # ★ 部署用 generator_ema（ema_decay=0.99 的平滑权重），非 raw generator（DMD 训练瞬态，长 rollout 会发散成噪声）。
        #   与已验证的 scripts/val/render_dmd2.py 一致：sd.get("generator_ema") or sd["generator"]。
        gen_sd = sd.get("generator_ema") or sd.get("generator") or sd
        which = "generator_ema" if "generator_ema" in sd else "generator"
        dmd.generator.load_state_dict(gen_sd, strict=False)
        print(f"[gen] loaded {gen_ckpt} [{which}]")
    dmd.generator.eval().requires_grad_(False)                          # 推理：渲染器冻结

    # ★ DMD2Models.__init__ 是训练/部署两用的共享构造函数，不管调用方要不要都会把 teacher(real_score,
    #   冻结双向,~1.7B)/critic(fake_score,~1.7B)/disc_head 建出来——这三个只在 DMD 训练的损失计算里
    #   用得到，VideoStream.render_block/decode_pixels 从头到尾只碰 dmd.generator，训练组件对"真实
    #   部署"纯粹是几个GB的显存浪费(之前只 dmd.critic.eval() 没有真正释放, 显存占用因此虚高)。这里建完
    #   即删；DMD2Models.set_reference() 已经用 getattr 兜底跳过缺失的 r_critic/r_teacher（见 models.py），
    #   训练脚本不受影响(那边这几个属性都在)。
    del dmd.teacher, dmd.critic, dmd.disc_head, dmd.r_critic, dmd.r_teacher
    dmd._mid_feat = None
    torch.cuda.empty_cache()

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(device, dtype).eval()
    vae.requires_grad_(False)

    # 注：compile 必须配 causal_window>0（shape 静态）才有效——window=0 时 KV 无界增长会 recompile 风暴。
    # reduce-overhead 用 CUDA 图，实测隔离渲染稳定(x1.37)；若在双线程 run() + 独立 vae_stream 下遇到
    # 图捕获冲突，把 mode 换成 "default"（少量收益换稳健）。首块编译预热 ~3-5min，持久化服务值得开。
    if compile_gen:
        try:
            dmd.generator = torch.compile(dmd.generator, dynamic=True, fullgraph=False,
                                          mode="reduce-overhead")
            print("[compile] generator compiled (dynamic, reduce-overhead)")
        except Exception as e:
            print(f"[compile] generator compile FAILED, fallback eager: {e}")
    if compile_vae:
        try:
            vae.decode = torch.compile(vae.decode, fullgraph=False, mode="reduce-overhead")
            print("[compile] vae.decode compiled")
        except Exception as e:
            print(f"[compile] vae compile FAILED, fallback eager: {e}")

    vae_stream = torch.cuda.Stream() if (overlap_vae and device.type == "cuda") else None
    mstream = MotionStream(motion_model, device, dtype, **(motion_cfg or {}))
    vstream = VideoStream(dmd, vae, device, dtype,
                          denoising_step_list=denoising_step_list, block_size=block_size,
                          vae_stream=vae_stream)
    return DualStreamPipeline(mstream, vstream)


if __name__ == "__main__":
    # 冒烟：随机条件跑通 shape。真实使用时把 text/audio/local_audio/ref_* 换成预处理结果。
    import os
    dev = torch.device(os.environ.get("SMOKE_DEV", "cuda:0"))
    dt = torch.bfloat16
    import sys; sys.path.append("/media/ps/ssd5/ayr/motar")
    from model.armodel import MotionTransformer

    seq_len = 64                                                        # 8 个 block
    mm = MotionTransformer(motion_dim=512, audio_dim=768, depth=8, heads=8,
                           dim_head=64, max_len=128, diffloss_dim=512,
                           diffloss_depth=2, num_sampling_steps=50).to(dev, dt).eval()
    # mm.load_pretrained("<ar_ckpt>.pt")  # 实测请加载权重 + set_latent_stats

    pipe = build_pipeline(mm, dev, dt,
                          gen_ckpt="/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt",
                          block_size=8)

    B = 1
    text_emb = torch.randn(B, 16, 768, device=dev, dtype=dt)
    audio_emb = torch.randn(B, 16, 768, device=dev, dtype=dt)
    local_audio = torch.randn(B, seq_len, 5, 768, device=dev, dtype=dt)
    ref_latent = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)        # VAE 编码后的参考帧潜码
    ref_img = torch.randn(1, 3, 224, 224, device=dev, dtype=dt)         # CLIP 输入的参考图

    t0 = time.perf_counter()
    n = 0
    for pix in pipe.run(text_emb, audio_emb, local_audio, ref_latent, ref_img, seq_len):
        n += pix.shape[0]
        print(f"[block] pixels {tuple(pix.shape)}  累计 {n} 帧  "
              f"耗时 {(time.perf_counter()-t0):.2f}s")
    print(f"done: {n} 帧 / {(time.perf_counter()-t0):.2f}s = {n/(time.perf_counter()-t0):.1f} fps")

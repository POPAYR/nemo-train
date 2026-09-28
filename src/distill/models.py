"""
Phase 3: DMD2 三网络封装（DECODER_DISTILL_PLAN.md §Phase 3）
============================================================
3 个去噪 UNet（都 ε-pred）：
  - generator（因果 student）：TemporalCausalControl(stream) + ode_8000 + LoRA(spatial/motion) + temporal 全训
  - fake_score / critic（双向）：teacher 基座 + LoRA
  - real_score / teacher（双向）：原始冻结
共享 1 个冻结 reference UNet + bank（同 ref→同身份特征，3 个 reader 各自读）。
统一 `(eps, x0)` 接口。
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from diffusers import DDIMScheduler
from transformers import CLIPVisionModelWithProjection

from ..models.unet_2d_condition import UNet2DConditionModel
from ..models.unet_3d import UNet3DConditionModel
from ..models.mutual_self_attention import ReferenceAttentionControl
from ..models.temporal_causal import TemporalCausalControl
from .dmd_loss import eps_to_x0
from .flow_math import v_to_x0

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
LORA_TARGETS = ["to_q", "to_k", "to_v"]    # 注意力投影；spatial+motion+temporal 都会命中


class DiscHead(nn.Module):
    """GAN 判别器头（官方 gan.py 迁移；DiT 深层特征→UNet mid-block 瓶颈特征）。
    输入 critic mid-block 特征 [B,C=1280,F,h,w] → 逐帧 logit [B,F]（空间池化）。
    train-only：推理时丢弃，不影响 d8 实时目标。"""
    def __init__(self, in_ch=1280, hidden=512, groups=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, hidden, 1),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, feat):               # feat [B,C,F,h,w]（bf16，与 critic 同 dtype→ZeRO 要求一致）
        x = self.net(feat)                 # gan loss 里再 .float()（softplus 前）
        return x.mean(dim=[3, 4]).squeeze(1)   # [B,F] 逐帧 logit


def _fresh_unet(cfg, ic, dev, dt):
    u = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    u.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    u.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    return u


def add_lora(unet, rank=16, alpha=16, full_temporal=False, grad_ckpt=True):
    """对注意力投影加 LoRA（base 冻结）；full_temporal=True 时再把 temporal_modules base 全解冻。"""
    unet.requires_grad_(False)
    if grad_ckpt:
        try: unet.enable_gradient_checkpointing()
        except Exception as e: print("[warn] grad ckpt:", e)
    lcfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none")
    unet = get_peft_model(unet, lcfg)
    if full_temporal:
        for n, p in unet.named_parameters():
            if "temporal_modules" in n and "lora_" not in n:
                p.requires_grad_(True)
    return unet


class DMD2Models:
    def __init__(self, dev, dt=torch.bfloat16, gen_ckpt=None, block_size=8,
                 lora_rank=16, real_guidance_scale=2.5, objective="eps", flow_ckpt=None):
        cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
        self.dev, self.dt, self.cfg = dev, dt, cfg
        self.real_guidance_scale = real_guidance_scale
        self.block_size = block_size
        # objective="eps"(原 DDPM ε-pred,不变) / "flow"(rectified-flow v-pred teacher)
        self.objective = objective
        self.scheduler = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
        self.acp = self.scheduler.alphas_cumprod.to(dev)
        self.num_train = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))

        # 共享冻结组件
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval().requires_grad_(False)
        self.reference_unet.load_state_dict(
            torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path).to(dev, dt).eval().requires_grad_(False)

        # flow 模式:把 rectified-flow v-pred teacher 权重灌进 teacher/critic/generator 三个基座。
        # flow_ckpt 顶层 key "denoising_unet" 是完整 UNet3D state_dict(spatial+temporal 已融合,1778 param)。
        _flow_sd = None
        if self.objective == "flow":
            assert flow_ckpt is not None, "objective=flow 需传 flow_ckpt"
            _raw = torch.load(flow_ckpt, map_location="cpu")
            _flow_sd = _raw["denoising_unet"] if "denoising_unet" in _raw else _raw
            print(f"[flow] teacher/critic/generator 基座 ← {flow_ckpt} "
                  f"(step={_raw.get('step')} obj={_raw.get('objective')})")

        # teacher（real_score）：冻结双向
        self.teacher = _fresh_unet(cfg, ic, dev, dt)
        if _flow_sd is not None:
            self.teacher.load_state_dict(_flow_sd, strict=False)
        self.teacher = self.teacher.eval().requires_grad_(False)

        # critic（fake_score）：teacher 基座【全量微调】双向（native: full finetune, 不用 LoRA）
        self.critic = _fresh_unet(cfg, ic, dev, dt)
        if _flow_sd is not None:
            self.critic.load_state_dict(_flow_sd, strict=False)
        self.critic.requires_grad_(True); self.critic.enable_gradient_checkpointing(); self.critic.train()

        # generator：因果 student。flow 模式从 flow teacher 起始(而非 ode_init);ε 模式从 ode_8000。
        # （.train() 而非 eval：XNeMo grad-ckpt 依赖 self.training，且 dropout=0 故等价）
        gen = _fresh_unet(cfg, ic, dev, dt)
        if _flow_sd is not None:
            gen.load_state_dict(_flow_sd, strict=False)
        if gen_ckpt:
            sd = torch.load(gen_ckpt, map_location="cpu")
            gen.load_state_dict(sd["denoising_unet"] if "denoising_unet" in sd else sd, strict=True)
            print(f"[gen] loaded {gen_ckpt}")
        gen.requires_grad_(True); gen.enable_gradient_checkpointing()
        self.generator = gen.train()
        self.gen_causal = TemporalCausalControl(self.generator, block_size=block_size, window=0)

        # ★ RoPE 接线:flow 血统(CUM1500 / 因果 init)的 temporal 权重是配 RoPE 训出来的,
        #   不开就会走原来的加性绝对 PE —— 权重与位置编码错配,**静默产出垃圾**(不会报错)。
        #   teacher/critic 是双向打分器 → bidir + RoPE;generator 是因果 student → 由
        #   gen_causal 统一管 mode,这里只置 RoPE 开关。
        self.rope = bool(_raw.get("rope", False)) if self.objective == "flow" else False
        if self.rope:
            from ..models.temporal_causal import set_temporal_rope
            n_t = set_temporal_rope(self.teacher, True, mode="bidir")
            n_c = set_temporal_rope(self.critic, True, mode="bidir")
            self.gen_causal.set_rope(True)
            print(f"[rope] 已启用:teacher {n_t} 层 / critic {n_c} 层 / generator {len(self.gen_causal)} 层(因果)")
        elif self.objective == "flow":
            print("[rope] ⚠️ flow_ckpt 未标记 rope=True,按加性绝对 PE 运行 —— 若权重是 RoPE 训的,结果会是垃圾")

        # 每个去噪 UNet 一个 reader（共享同一个 writer 的 bank）
        _rac = lambda net, mode: ReferenceAttentionControl(
            net, do_classifier_free_guidance=False, mode=mode, batch_size=1, fusion_blocks="full")
        self.writer = _rac(self.reference_unet, "write")
        self.r_gen = _rac(self.generator, "read")
        self.r_critic = _rac(self.critic, "read")
        self.r_teacher = _rac(self.teacher, "read")

        # ---- GAN 判别器头（DMD2）：挂在 critic 主干上，读 mid-block(瓶颈) 特征 → 逐帧 logit ----
        # 官方判别器 = fake_score 主干 + cls 头；我们复用 critic，只额外加这一个轻量头。
        # forward hook 抓 mid_block 输出；仅在 _capture_mid=True 时缓存（避免 DMD 前向白留大激活）。
        mid_ch = self.critic.config.block_out_channels[-1]     # 1280
        self.disc_head = DiscHead(in_ch=mid_ch).to(dev, dt)    # bf16，与 critic 同 dtype（ZeRO 要求全同 dtype）
        self._mid_feat = None
        self._capture_mid = False

        def _mid_hook(_mod, _inp, out):
            if self._capture_mid:
                self._mid_feat = out[0] if isinstance(out, (tuple, list)) else out
        self.critic.mid_block.register_forward_hook(_mid_hook)

        for tag, m in [("teacher(frozen)", self.teacher), ("critic", self.critic),
                       ("generator", self.generator), ("disc_head", self.disc_head)]:
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"[{tag}] trainable={n/1e6:.1f}M")

    @torch.no_grad()
    def set_reference(self, ref_latent, clip_emb, batch_size):
        """跑一次 reference UNet，把身份特征 bank 分发给 3 个 reader。
        ★ r_critic/r_teacher 在纯推理部署场景(见 stream_pipeline.py::build_pipeline)会被主动 del 掉
        （teacher/critic 是 DMD 训练专用组件，推理只用 generator，留着白占几个GB显存），
        这里用 getattr 兜底跳过缺失的 reader，训练脚本(三个 reader 都在)行为不变。"""
        self.writer.batch_size = batch_size
        self.writer.clear()
        self.reference_unet(ref_latent, torch.zeros((), device=self.dev).long(),
                            encoder_hidden_states=clip_emb, return_dict=False)
        for r in (getattr(self, "r_gen", None), getattr(self, "r_critic", None), getattr(self, "r_teacher", None)):
            if r is None:
                continue
            r.batch_size = batch_size
            r.update(self.writer, dtype=self.dt)

    def clip_embed(self, ref_img):
        return self.image_encoder(ref_img.to(self.dev, self.dt)).image_embeds.unsqueeze(1)

    def forward_net(self, net, x_t, t, clip_emb, motion, capture_disc=False):
        """net∈{generator,critic,teacher}；返回 (eps, x0)。t: [B] 或标量。
        capture_disc=True（仅 net=critic）：同一次前向抓 mid-block 特征 → 额外返回逐帧 logit [B,F]。"""
        if capture_disc:
            assert net is self.critic, "disc logit 只能从 critic 主干出"
            self._capture_mid = True
        eps = net(x_t, t, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
        x0 = eps_to_x0(x_t, eps, t.reshape(-1) if torch.is_tensor(t) else t, self.acp)
        if capture_disc:
            self._capture_mid = False
            logit = self.disc_head(self._mid_feat); self._mid_feat = None
            return eps, x0, logit
        return eps, x0

    def forward_net_flow(self, net, z, sigma, clip_emb, motion, capture_disc=False):
        """flow 版前向:net∈{generator,critic,teacher} 出速度 v,反演 x0 = z − σ·v。
        z: 带噪 latent z_σ。sigma: [B] 或标量(∈[0,1])。网络 timestep 输入 = σ·(num_train-1)。
        返回 (v, x0)[, logit]。capture_disc=True(仅 critic)额外返回逐帧 logit。"""
        s = sigma if torch.is_tensor(sigma) else torch.full((z.shape[0],), float(sigma), device=z.device)
        s = s.reshape(-1).to(z.device)
        t_emb = (s * self.num_train).to(self.dt)                      # σ·1000(对齐 flow_teacher_ft)
        if capture_disc:
            assert net is self.critic, "disc logit 只能从 critic 主干出"
            self._capture_mid = True
        v = net(z, t_emb, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
        x0 = v_to_x0(z, v, s)
        if capture_disc:
            self._capture_mid = False
            logit = self.disc_head(self._mid_feat); self._mid_feat = None
            return v, x0, logit
        return v, x0

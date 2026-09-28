"""
AR motion 训练：DMD-through-G —— 视频空间分布匹配蒸馏 motion（替代 GAN）。
=====================================================================================
= VSD/DMD 机制迁到 audio→motion（见 motar/AR_MOTION_WORK_SUMMARY.md §10）。
- Generator = motion model（训，DeepSpeed）。
- Renderer  = 已蒸 4 步因果视频解码器（冻结；把 m̂ 渲成 x̂ 供打分）。
- s_real    = 未蒸 teacher（冻结，单次前向 | m_gt）。
- s_fake    = critic（训，手动 AdamW，单次前向 | m̂）。
loss: L = λ_mse·L_mse(motion 空间锚) + w_dmd·L_DMD-through-G(视频空间分布匹配)。
长 rollout(128) → 随机 24 帧窗 → 蒸馏解码器渲染 → DMD 梯度穿渲染器回 motion。
继承 XNemoARTrainer 复用数据/config/可视化/DeepSpeed/rollout，只重写模型构造+train step+存档。
"""
import os
import sys
import argparse
import datetime
import random
import torch
import torch.distributed as dist
import deepspeed
from omegaconf import OmegaConf

XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path:
    sys.path.append(XNEMO_ROOT)

from scripts.train.train_ar_xnemo import XNemoARTrainer, rollout_lcm
from src.losses import motion_regression_loss, std_ratio
from src.distill.models import DMD2Models
from src.distill.rollout import self_forcing_rollout
from src.distill.dmd_through_g import generator_step, critic_step


class DMDThroughGTrainer(XNemoARTrainer):

    # ---------------- 模型：super 建 motion/encoders/teacher/data + 额外视频三件套 ----------------
    def _init_models(self):
        super()._init_models()
        cfg = self.config
        dcfg = cfg.dmd_g
        self.w_dmd = float(dcfg.get("w_dmd", 1.0))
        self.lam_mse_dmd = float(dcfg.get("lam_mse", 1.0))
        self.dmdg_window = int(dcfg.get("window", 24))
        self.dmdg_block = int(dcfg.get("block_size", 8))
        self.dmdg_dsl = [int(x) for x in dcfg.get("dsl", [999, 749, 499, 249])]
        self.critic_grad_clip = float(dcfg.get("critic_grad_clip", 10.0))

    def _setup_deepspeed(self):
        import json
        cfg = self.config
        # ⭐ 多卡 deepspeed launcher 下传 config 文件路径(str)会被误当 base64 解码 → 直接传 dict
        ds_dict = json.load(open(str(cfg.deepspeed_config_path)))
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            args=self.args, model=self.model, model_parameters=None, config=ds_dict)
        self.disc_optimizer = None
        self.start_epoch, self.start_step = self._resume_epoch, self._resume_step
        self.dtype = torch.bfloat16 if self.model_engine.bfloat16_enabled() else torch.float32
        self.denoising_unet.enable_gradient_checkpointing()
        if hasattr(self.motion_predictor, "enable_gradient_checkpointing"):
            self.motion_predictor.enable_gradient_checkpointing()

        dcfg = self.config.dmd_g
        # DMD2Models 一站式：generator(渲染器) + teacher(s_real) + critic(s_fake) + reference + 前向接口
        self.Mvid = DMD2Models(self.device, dt=self.dtype, gen_ckpt=None, block_size=self.dmdg_block)
        sd = torch.load(dcfg.render_ckpt, map_location="cpu")
        self.Mvid.generator.load_state_dict(sd["generator"], strict=True)
        self.Mvid.generator.requires_grad_(False)          # 渲染器冻结（梯度只穿过它回 motion，不训它）
        # ⭐ 方案②：s_real=ref-only score(载入 teacher 位、冻结)、s_fake=critic 从 s_real init
        ro = torch.load(dcfg.refonly_ckpt, map_location="cpu")["critic"]
        self.Mvid.teacher.load_state_dict(ro, strict=True)   # s_real = ref-only（teacher 位，已冻结）
        self.Mvid.critic.load_state_dict(ro, strict=True)    # s_fake 从 s_real init
        rc = dcfg.get("resume_critic_path", None)
        if rc and os.path.exists(rc):
            self.Mvid.critic.load_state_dict(torch.load(rc, map_location="cpu")["critic"], strict=True)
        # motion_encoder：算打分用的 neg_motion(中性 motion)
        from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
        self.mot_enc = MotEncoder().to(self.device, self.dtype).eval().requires_grad_(False)
        self.mot_enc.load_state_dict(torch.load(self.config.motion_encoder_path, map_location="cpu"), strict=True)
        self.critic_params = [p for p in self.Mvid.critic.parameters() if p.requires_grad]
        self.critic_opt = torch.optim.AdamW(self.critic_params, lr=float(dcfg.get("critic_lr", 4e-7)),
                                            betas=(0.0, 0.999), weight_decay=float(dcfg.get("critic_wd", 0.01)))
        if self.is_main():
            n_c = sum(p.numel() for p in self.critic_params) / 1e6
            print(f"[DMD-G] renderer←{os.path.basename(dcfg.render_ckpt)}  s_real/s_fake←{os.path.basename(dcfg.refonly_ckpt)}  "
                  f"critic trainable={n_c:.1f}M w_dmd={self.w_dmd} lam_mse={self.lam_mse_dmd}")

    def _run_dwarmup_phase(self, global_step):
        return global_step        # DMD-through-G 无 GAN warmup

    def _prepare_data(self):
        # DMD 只需 ref 单帧(不需完整 GT 视频序列)→ pad_short=True(pad+mask)让 L<frames 的短样本也可用
        from data.dataset import MotarDataset
        from torch.utils.data import ConcatDataset, DataLoader
        from torch.utils.data.distributed import DistributedSampler
        cfg = self.config
        def make(src):
            return MotarDataset(
                pose_dir=src.pose_dir, audio_dir=src.audio_dir, caption_dir=src.caption_dir,
                data_name_path=src.data_name_path, tokenizer_path=cfg.data.tokenizer_path,
                data_stats_path=cfg.data.data_stats_path, context_length=cfg.data.frames,
                fps=cfg.data.fps, sr=cfg.data.sr, text_max_len=cfg.data.get("text_max_len", 128),
                random_crop=True, pad_short=True, load_video=True,
                latent_dir=src.latent_dir, video_dir=src.video_dir, video_processor=self.video_processor)
        ds = [make(s) for s in cfg.data.sources]
        self.train_dataset = ConcatDataset(ds) if len(ds) > 1 else ds[0]
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=cfg.data.batch_size, sampler=self.train_sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=cfg.data.get("prefetch_factor", 3), persistent_workers=True)
        self.dwarm_dataloader = None

    def _denorm(self, x):
        return x * (self.std.reshape(1, 1, -1) + 1e-6) + self.mean.reshape(1, 1, -1)

    # ------------------------------- 一步 ------------------------------- #
    def train_one_step(self, batch, global_step):
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        B, T = batch["motion_tensor"].shape[:2]
        gt_motion = batch["motion_tensor"]                  # [B,T,D] 归一化空间
        frame_mask = batch.get("mask", None)
        text_emb, audio_emb, local_audio_emb, clip_img_emb = self._encode_conditions(batch, B, T)
        clip_img_emb = clip_img_emb.to(self.dtype)      # 喂 bf16 视频 UNet，避免 dtype 不匹配

        with torch.enable_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            # 1) 长 rollout（128 帧，带梯度）→ m̂
            preds = rollout_lcm(self.motion_predictor, gt_motion, text_emb, audio_emb, local_audio_emb,
                                self.alphas_cp, self.T_max,
                                autocast_factory=lambda: torch.cuda.amp.autocast(dtype=self.dtype))
            m_hat = torch.cat([gt_motion[:, 0:1].detach(), preds], dim=1)   # [B,T,D]

            # 2) motion 空间锚 L_mse（全长 128）
            l_mse = motion_regression_loss(m_hat, gt_motion, frame_mask=frame_mask,
                                           kind=self.config.loss.get("reg_kind", "smooth_l1"))

            # 3) 随机连续 24 帧窗：**只在有效帧内取**（避开 pad 帧，否则渲染静止 pad → DMD 瞎打分）
            W = self.dmdg_window
            valid = int(frame_mask[0].sum().item()) if frame_mask is not None else T
            valid = max(W, min(valid, T))
            k = int(torch.randint(0, valid - W + 1, (1,)).item())
            m_hat_tok = self._denorm(m_hat[:, k:k + W]).reshape(B, W, 32, 16).to(self.dtype)   # 渲染用(带梯度)
            # ⭐ 方案②：打分 motion = neg_motion(中性,motion_encoder 编码参考帧)，不是 m̂ →
            #    s_real 判"真实说话视频分布"(动态)，塌 motion 渲的静态 x̂ 被判低密度 → 顶 std。
            rmc = batch["ref_mot_cond"].to(self.device, self.dtype)
            bbox = torch.ones((B, 3), device=self.device, dtype=self.dtype); bbox[:, :2] = 0
            with torch.no_grad():
                neg = self.mot_enc(rmc, bbox)
            neg_mot = neg.unsqueeze(1).expand(B, W, *neg.shape[1:]).to(self.dtype)   # [B,W,32,16]

            # 4) 渲染器：蒸馏解码器可微渲 W 帧(条件 m̂)；5) DMD 打分（s_real/s_fake 同点 x̂、条件 neg_motion）
            ref_latent = (batch["ref_latent"] * self.vae_scale).to(self.dtype)
            self.Mvid.set_reference(ref_latent, clip_img_emb, B)

            def render_fn():
                noise = torch.randn(B, 4, W, 64, 64, device=self.device, dtype=self.dtype)
                x0, _ = self_forcing_rollout(self.Mvid, noise, clip_img_emb, m_hat_tok, self.dmdg_dsl,
                                             block_size=self.dmdg_block, grad_window=W, full_steps=True)
                return x0
            s_real_fn = lambda x_t, t: self.Mvid.forward_net(self.Mvid.teacher, x_t, t, clip_img_emb, neg_mot)[1]
            s_fake_fn = lambda x_t, t: self.Mvid.forward_net(self.Mvid.critic, x_t, t, clip_img_emb, neg_mot)[1]
            s_fake_eps = lambda x_t, t: self.Mvid.forward_net(self.Mvid.critic, x_t, t, clip_img_emb, neg_mot)[0]

            l_dmd, x_hat, dmdlog = generator_step(render_fn, s_real_fn, s_fake_fn, self.Mvid.scheduler, dtype=self.dtype)
            l_g = self.lam_mse_dmd * l_mse + self.w_dmd * l_dmd

        if torch.isnan(l_g):
            return None, None, "Loss NaN"

        # 6) 更新 Generator（DeepSpeed；梯度穿冻结渲染器回 motion）
        self.model_engine.backward(l_g)
        self.model_engine.step()

        # 7) 更新 critic（s_fake，手动 AdamW，隔离于 DeepSpeed）
        self.critic_opt.zero_grad(set_to_none=True)
        l_c, _ = critic_step(x_hat.detach(), s_fake_eps, self.Mvid.scheduler, dtype=self.dtype)
        l_c.backward()
        gn_c = torch.nn.utils.clip_grad_norm_(self.critic_params, self.critic_grad_clip)
        self.critic_opt.step()

        logs = {
            "loss": l_g.item(), "l_dmd": l_dmd.item(), "l_mse": l_mse.item(),
            "l_critic": l_c.item(), "critic_gn": float(gn_c), "x0_std": dmdlog["x0_std"],
            "std_ratio": std_ratio(m_hat.detach(), gt_motion), "win_k": float(k),
            "lam_video": 0.0, "lam_reg": self.lam_mse_dmd, "lam_adv": self.w_dmd,  # fit() 日志兼容占位
        }
        # ---- 累积 log_every 步再平均（仿 sf_gan_video.py 的 run 累积器）----
        # micro=1 → 每步是单样本瞬时值，方差极大(std_ratio 0.3~1.4 乱跳)，必须跨步平均才看得出趋势。
        KEYS = ("l_dmd", "l_mse", "l_critic", "x0_std", "critic_gn", "std_ratio")
        if not hasattr(self, "_run"):
            self._run = {k: 0.0 for k in KEYS}; self._run["n"] = 0
        r = self._run
        r["l_dmd"] += l_dmd.item(); r["l_mse"] += l_mse.item(); r["l_critic"] += l_c.item()
        r["x0_std"] += dmdlog["x0_std"]; r["critic_gn"] += float(gn_c)
        r["std_ratio"] += logs["std_ratio"]; r["n"] += 1
        if self.is_main() and global_step % self.log_every == 0 and r["n"] > 0:
            n = r["n"]
            print(f"[DMD] step{global_step} l_dmd={r['l_dmd']/n:.4f} l_mse={r['l_mse']/n:.4f} "
                  f"l_critic={r['l_critic']/n:.4f} x0std={r['x0_std']/n:.3f} "
                  f"critic_gn={r['critic_gn']/n:.3f} std_ratio={r['std_ratio']/n:.4f} (avg/{n})")
            w = getattr(self, "writer", None)
            if w is not None:
                for kk in KEYS:
                    w.add_scalar(f"dmd/{kk}", r[kk] / n, global_step)
        if global_step % self.log_every == 0:      # 所有 rank 都要重置累积器
            self._run = {k: 0.0 for k in KEYS}; self._run["n"] = 0
        return logs, None, None

    # ------------------------------- 存档：motion + critic ------------------------------- #
    def _save_ar_checkpoint(self, epoch, step):
        params = list(self.motion_predictor.parameters())
        stage = 0
        try:
            stage = self.model_engine.zero_optimization_stage()
        except Exception:
            pass

        def _grab(m):
            return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

        if stage == 3:
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                msd = _grab(self.motion_predictor) if self.is_main() else None
        else:
            msd = _grab(self.motion_predictor) if self.is_main() else None

        if self.is_main():
            ck = {"motion_predictor": msd, "critic": _grab(self.Mvid.critic),
                  "step": step, "epoch": epoch,
                  "critic_opt": self.critic_opt.state_dict()}
            path = self.ckpt_dir / f"dmdg_step_{step}.pt"
            torch.save(ck, path)
            # 单独再存一份轻量 critic（resume 用，避免和 motion 混）
            torch.save({"critic": ck["critic"], "step": step},
                       self.ckpt_dir / "critic_latest.pt")
            print(f"[save] step {step} → {path} ({os.path.getsize(path)/1e9:.1f}GB)")
            # 只保留最近 keep_last 个
            keep = int(self.config.training.get("keep_last", 3))
            olds = sorted(self.ckpt_dir.glob("dmdg_step_*.pt"),
                          key=lambda p: int(p.stem.split("_")[-1]))[:-keep]
            for o in olds:
                try: o.unlink()
                except Exception: pass
        if dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    deepspeed.init_distributed()
    config = OmegaConf.load(args.config)
    trainer = DMDThroughGTrainer(config, args)
    trainer.fit()

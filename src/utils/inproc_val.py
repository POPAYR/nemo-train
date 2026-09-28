"""训练循环内的验证（CLAUDE.md §6.4）。

设计要点
--------
1. **固定验证集**：从 `eval_metrics/testset` 取固定的 clip + 固定帧号 + 固定噪声种子，
   保证不同 step、不同 run 之间**逐点可比**。绝不用随机样本——那样看到的是采样方差不是训练进展。
2. **两档指标**：
   - `vmse`：逐 σ 的 v-MSE，不做采样，**极便宜**（一次前向/σ），每次验证都算。
     ⚠️ 它很钝（实测 v-MSE 变 0.3% ↔ FVD 变 5.7%），只能看趋势不能定优劣。
   - `sample`：真的采样 + VAE 解码 + PSNR/SSIM/LPIPS/FID，贵但准。按 `--val_sample_every` 稀疏跑。
3. **只在 rank0 跑，其余 rank 在 barrier 等**。验证集很小（默认 8 clip），
   分片 all_reduce 的复杂度不值得，且 rank0 单跑结果完全确定。
4. **产物落盘**：`<out>/eval/step_<N>.json`（结构化）+ `<out>/samples/step_<N>/*.png`（可视化）
   + TensorBoard 标量。人读的结论写 `<out>/eval/README.md` 由人补。

用法（训练脚本里）::

    from src.utils.inproc_val import Validator
    val = Validator(out_dir=args.out, kind="image", sigmas=(0.3,0.5,0.7,0.9),
                    n_clips=8, n_frames=4, sample_steps=20, is_main=is_main)
    ...
    if step % args.val_every == 0:
        val.run(step, denu_core, refu, imgenc, dev, dt,
                do_sample=(step % args.val_sample_every == 0))
"""
import os, json, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
import sys as _sys
if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party, remap  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
import numpy as np
import torch

TESTSET = XP("XN_TESTSET")
# ★ 可用环境变量 VAL_POSE_DIR 切换 motion latent 来源
#   (验证 bbox_proj 通路时要对比"常量 bbox"与"真实逐帧 bbox"两版)
# ★ 默认改成 pose_embed_real:fixed 裁剪重做后只产出真实逐帧 bbox 的那一份,
#   常量 bbox 的 pose_embed 已不再生成(bbox_drop 定为 0,它没有使用者)。
#   仍按目录存在与否兜底,避免旧数据环境下反而找不到。
_PB = XP("XN_HALLO3")
POSE = {"hallo3": os.environ.get(
    "VAL_POSE_DIR",
    f"{_PB}/pose_embed_real" if os.path.isdir(f"{_PB}/pose_embed_real")
    else f"{_PB}/pose_embed")}
LAT = {"hallo3": XP("XN_HALLO3", "frame_latent")}


class Validator:
    def __init__(self, out_dir, kind="image", ds="hallo3", n_clips=8, n_frames=4,
                 sigmas=(0.3, 0.5, 0.7, 0.9), sample_steps=20, sample_shift=1.0,
                 seed=1234, is_main=True, tb=None, max_frames=128, clip_len=64, val_cfg=2.0, val_window=0, val_overlap=4, val_ctx="official", shard=False,
                 student_sigmas=None, block=8, ctrl=None, set_ref_fn=None):
        """kind: "image"(stage1,逐帧独立) | "video"(stage2,整段双向) | "causal"(因果少步学生)

        ★ causal 与前两者的关键差别:采样必须用**训练一致的 renoise**
          (预测 x̂0 → 新鲜噪声加噪到下一档),不是 Euler。
          文档 §2.8:两者不等价,少步 Euler 会把 FVD 抬高一倍。
        """
        self.__dict__.update(locals()); del self.self
        self.dir_eval = os.path.join(out_dir, "eval")
        self.dir_samp = os.path.join(out_dir, "samples")
        if is_main:
            os.makedirs(self.dir_eval, exist_ok=True); os.makedirs(self.dir_samp, exist_ok=True)
        self.items = None; self._lpips = None; self._vae = None

    # ---------------------------------------------------------------- 数据
    def _dist(self):
        """返回 (rank, world) 当且仅当 shard=True 且分布式已初始化;否则 None。"""
        if not self.shard:
            return None
        import torch.distributed as _d
        if not (_d.is_available() and _d.is_initialized()) or _d.get_world_size() < 2:
            return None
        return _d.get_rank(), _d.get_world_size()

    def _load(self, dev, dt):
        if self.items is not None:
            return self.items
        man = json.load(open(f"{TESTSET}/manifest.json"))[self.ds]
        # ★ VAL_CLIPS_FILE:把验证集换成任意 clip 清单(每行一个)。
        #   用途:过拟合诊断实验必须在**训练用的那几条**上验证,否则测的是泛化不是记忆。
        #   这些 clip 不在测试集 manifest 里,故 n_frames/ref/gt_dir 需就地推导。
        _cf = os.environ.get("VAL_CLIPS_FILE")
        if _cf:
            names = [l.strip() for l in open(_cf) if l.strip()][: self.n_clips]
            _fb = XP("XN_HALLO3")
            man = dict(man)
            for c in names:
                if c in man:
                    continue
                gd = f"{_fb}/face_frames/{c}"
                nf = len([f for f in os.listdir(gd) if f.endswith(".jpg")])
                man[c] = {"n_frames": nf, "gt_dir": gd,
                          "ref": os.path.join(gd, sorted(os.listdir(gd))[0])}
            clips = names
        else:
            sel = set(json.load(open(f"{TESTSET}/{self.ds}_subset30.json")))
            clips = sorted(c for c in man if c in sel)[: self.n_clips]
        out = []
        for ci, c in enumerate(clips):
            m = man[c]
            lat = torch.load(f"{LAT[self.ds]}/{c}.pt", map_location="cpu").float()
            mo = torch.load(f"{POSE[self.ds]}/{c}.pt", map_location="cpu").float().reshape(-1, 32 * 16)
            F_ = min(m["n_frames"], len(lat), self.max_frames)
            if self.kind in ("video", "causal"):
                if F_ < self.clip_len: continue
                idx = np.arange(self.clip_len)                     # 整段连续,temporal 需要
            else:
                # ★ 跳过 t=0(=参考帧本身,所有 ckpt 都一样,无区分度),取靠后的帧(gap 大=难)
                idx = np.linspace(F_ // 4, F_ - 1, self.n_frames).astype(int)
            x0 = lat[idx].permute(1, 0, 2, 3).unsqueeze(0).to(dev, dt)   # [1,C,N,H,W]
            g = torch.Generator(device="cpu"); g.manual_seed(self.seed + ci)
            out.append(dict(clip=c, idx=idx, x0=x0, gi=len(out),
                            ref_lat=lat[0:1].to(dev, dt), ref_path=remap(m["ref"]), gt_dir=remap(m["gt_dir"]),   # manifest 里是开发机绝对路径
                            mo=mo[idx].reshape(1, len(idx), 32, 16).to(dev, dt),
                            eps=torch.randn(tuple(x0.shape), generator=g).to(dev)))
        self.items = out
        return out

    def _clip_emb(self, imgenc, ref_path, dev, dt):
        """与 image_pair_dataset._process_clip_image 逐行一致(抗锯齿缩放 + 不再 rescale/resize)。

        imgenc=None 时自行懒加载 CLIP(ode_init/DMD 的训练循环里没有 image encoder,
        它们的 clip_emb 是预存在轨迹文件里的)。结果按 ref_path 缓存,只算一次。"""
        if ref_path in getattr(self, "_ce_cache", {}):
            return self._ce_cache[ref_path]
        if imgenc is None:
            if getattr(self, "_imgenc", None) is None:
                from transformers import CLIPVisionModelWithProjection
                from omegaconf import OmegaConf as _OC
                _c = _OC.load(XP("REPO", "configs/test_ar_model.yaml"))
                self._imgenc = CLIPVisionModelWithProjection.from_pretrained(
                    _c.image_encoder_path).to(dev, dt).eval()
            imgenc = self._imgenc
        from PIL import Image
        from diffusers.video_processor import VideoProcessor
        from transformers import CLIPImageProcessor
        from utils.processor import _resize_with_antialiasing
        vp = VideoProcessor(do_resize=True, vae_scale_factor=8)
        pil = Image.open(ref_path).convert("RGB").resize((512, 512))
        pt = vp.numpy_to_pt(vp.pil_to_numpy(pil)).squeeze(0).unsqueeze(0)
        pt = _resize_with_antialiasing(pt * 2.0 - 1.0, (224, 224))
        px = CLIPImageProcessor()(images=((pt + 1.0) / 2.0).squeeze(0), return_tensors="pt",
                                  do_rescale=False, do_resize=False).pixel_values
        ce = imgenc(px.to(dev, dt)).image_embeds.unsqueeze(1)
        if not hasattr(self, "_ce_cache"): self._ce_cache = {}
        self._ce_cache[ref_path] = ce
        return ce

    def _set_ref(self, refu, rr, rw, it, ce, dev, dt):
        """写 reference bank。DMD 那边 bank 由 DMD2Models.set_reference 统一分发给三个 reader,
        没有裸的 refu/rreader/rwriter,故用 set_ref_fn 回调接进来。"""
        if self.set_ref_fn is not None:
            self.set_ref_fn(it["ref_lat"], ce); return
        rw.clear()
        refu(it["ref_lat"], torch.zeros((), device=dev).long(),
             encoder_hidden_states=ce, return_dict=False)
        rr.update(rw, dtype=dt)

    # ---------------------------------------------------------------- 主入口
    @torch.no_grad()
    def run(self, step, denu, refu, imgenc, dev, dt, rr=None, rw=None, do_sample=False, logger=None):
        # ★ shard=True 且处于分布式:所有 rank 都进来,clip 按 rank 轮流分(items[rank::world]),
        #   各自采样后汇总到 rank0 算 FID/FVD 与写日志。否则保持旧行为:只 rank0 跑。
        _dd = self._dist()
        if not self.is_main and _dd is None:
            return None
        t0 = time.time()
        was_training = denu.training; denu.eval()
        items_all = self._load(dev, dt)
        items = items_all[_dd[0]::_dd[1]] if _dd else items_all
        res = {"step": step, "n_clips": len(items_all), "kind": self.kind}

        # ---- ① 便宜档
        # causal:按训练目标(每块独立档 → 预测 x̂0 → 与 GT 比)逐档算 x0-MSE;
        # image/video:逐 σ 算 v-MSE。
        sig_probe = list(self.student_sigmas) if self.kind == "causal" else list(self.sigmas)
        per_sigma = {}
        for _si, s in enumerate(sig_probe):
            print(f"[val:prog] step {step} 便宜档 σ={s} ({_si+1}/{len(sig_probe)})", flush=True)
            acc = []
            for it in items:
                ce = self._clip_emb(imgenc, it["ref_path"], dev, dt)
                self._set_ref(refu, rr, rw, it, ce, dev, dt)
                x0 = it["x0"].float(); eps = it["eps"]
                z = ((1 - s) * x0 + s * eps).to(dt)
                if self.kind == "causal":
                    Fn = z.shape[2]
                    t_emb = torch.full((1, Fn), s * 1000.0, device=dev, dtype=dt)  # 逐帧档
                    if self.ctrl is not None:
                        self.ctrl.set_mode("train"); self.ctrl.set_offset(0)
                    v = denu(z, t_emb, encoder_hidden_states=[ce, it["mo"]],
                             pose_cond_fea=None, return_dict=False)[0]
                    x0p = z.float() - s * v.float()
                    acc.append(float(((x0p - x0) ** 2).mean()))          # x0-MSE,与 ode_init 目标同形
                else:
                    v = denu(z, torch.full((1,), s * 1000.0, device=dev, dtype=dt),
                             encoder_hidden_states=[ce, it["mo"]], pose_cond_fea=None, return_dict=False)[0]
                    acc.append(float(((v.float() - (eps - x0)) ** 2).mean()))
            if _dd:
                _all = [None] * _dd[1]
                torch.distributed.all_gather_object(_all, acc)
                acc = [a for part in _all for a in part]
            per_sigma[f"{s}"] = float(np.mean(acc))
        key = "x0mse" if self.kind == "causal" else "vmse"
        res[key] = per_sigma
        res[key + "_mean"] = float(np.mean(list(per_sigma.values())))

        # ---- ② 采样 + 画质指标(贵,稀疏跑)
        if do_sample:
            res.update(self._sample_metrics(step, denu, refu, imgenc, dev, dt, rr, rw, items) or {})
        if _dd and not self.is_main:
            if was_training: denu.train()
            return None

        # ★ 必须复位因果控制器:_causal_rollout 结尾留下 commit=True / stream 模式 / 脏 cache,
        #   带进训练前向会改变计算图 → DDP 报「参数被标记 ready 两次」。
        #   单卡冒烟抓不到(无 DDP hook 检查),四卡才炸。
        if self.ctrl is not None:
            self.ctrl.set_mode("train"); self.ctrl.reset_cache()
            self.ctrl.set_commit(False); self.ctrl.set_offset(0)
        res["sec"] = round(time.time() - t0, 1)
        with open(os.path.join(self.dir_eval, f"step_{step:06d}.json"), "w") as f:
            json.dump(res, f, indent=2)
        if self.tb is not None:
            for k, v in per_sigma.items(): self.tb.add_scalar(f"val/{key}_sigma{k}", v, step)
            self.tb.add_scalar(f"val/{key}_mean", res[key + "_mean"], step)
            for k in ("PSNR", "SSIM", "LPIPS", "FID", "FVD", "DYN", "HFE", "TXF", "FLK"):
                if k in res: self.tb.add_scalar(f"val/{k}", res[k], step)
        msg = (f"[val] step {step}  {key}_mean={res[key + '_mean']:.5f}  "
               + " ".join(f"σ{k}={v:.4f}" for k, v in per_sigma.items())
               + (f"  | cfg{float(self.val_cfg):g} PSNR={res['PSNR']:.2f} SSIM={res['SSIM']:.4f} "
                  f"LPIPS={res['LPIPS']:.4f} FID={res['FID']:.2f}"
                  + (f" FVD={res['FVD']:.1f}" if res.get("FVD") else "")
                  + (f" DYN={res['DYN']:.3f}" if res.get("DYN") else "")
                  + (f" FLK={res['FLK']:.3f}" if res.get("FLK") else "") if "PSNR" in res else "")
               + f"  ({res['sec']}s)")
        (logger.info if logger else print)(msg)
        if was_training: denu.train()
        return res

    # ---------------------------------------------------------------- 采样分支
    def _sample_metrics(self, step, denu, refu, imgenc, dev, dt, rr, rw, items):
        import lpips as _lp
        from skimage.metrics import structural_similarity as ssim_fn
        from PIL import Image
        from einops import rearrange
        from diffusers import AutoencoderKLTemporalDecoder
        from omegaconf import OmegaConf
        if self._vae is None:
            cfg = OmegaConf.load(XP("REPO", "configs/test_ar_model.yaml"))
            self._vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
        if self._lpips is None:
            self._lpips = _lp.LPIPS(net="vgg").to(dev).eval()
        n = self.sample_steps
        t = torch.linspace(1.0, 0.0, n + 1, device=dev)
        sig = self.sample_shift * t / (1 + (self.sample_shift - 1) * t) if self.sample_shift > 1 else t
        sdir = os.path.join(self.dir_samp, f"step_{step:06d}"); os.makedirs(sdir, exist_ok=True)
        P, S, L = [], [], []
        VG, VR = [], []          # 视频类的整段帧,给 FVD/DYN/FLK 用
        sheet = []               # image 类的总览图:所有 clip 竖着摞成一张,一次看完
        gdir, rdir = os.path.join(sdir, "_fid_gen"), os.path.join(sdir, "_fid_gt")
        os.makedirs(gdir, exist_ok=True); os.makedirs(rdir, exist_ok=True)
        for _ci, it in enumerate(items):
            print(f"[val:prog] step {step} 采样 {_ci+1}/{len(items)} clip={it['clip'][:10]}", flush=True)
            ce = self._clip_emb(imgenc, it["ref_path"], dev, dt)
            self._set_ref(refu, rr, rw, it, ce, dev, dt)
            if self.kind == "causal":
                z = self._causal_rollout(denu, ce, it, dev, dt)
            else:
                z = it["eps"].to(dt).clone()             # ★ 复用固定噪声,跨 step 可比
                # ★ val_cfg:评测必须在**交付工作点**上做。原来固定 cfg=1.0,
                #   而交付用 1.2~1.5,导致两次跨线排序判反(见 DATA.md §22)。
                #   uncond 是三重置空(bank 不注入 + CLIP 置零 + 参考帧 motion),
                #   与 tool/render_s2_cfg.py 一致;只换 motion 是错的(历史教训)。
                w = float(self.val_cfg)
                mo_c = it["mo"]
                mo_u = mo_c[:, 0:1].expand_as(mo_c).contiguous()
                ce_u = torch.zeros_like(ce)
                # ★ val_window>0:用滑窗生成(L=24 模型必须如此,否则测的不是它的能力)。
                #   累加 + counter 归一,与 src/pipelines/pipeline_pose2vid_motenc_long.py 同构;
                #   重叠区被多个窗口覆盖,除以 counter 保证尺度正确。
                F_ = z.shape[2]
                # ★ 滑窗调度与 X-NeMo 官方一致(src/utils/ctx_sched.py):每步随机偏移 + 环形窗口。
                #   旧实现窗口固定不动,接缝永远落在同几帧,20 步误差累积成可见跳变(2026-09-23 用户肉眼发现)。
                #   val_ctx="fixed" 可复现旧行为。rng 按 clip 固定 seed,val 可复现。
                import random as _random
                from src.utils.ctx_sched import windows as _windows
                _rng = _random.Random(self.seed * 1000 + it.get("gi", _ci))

                def _pred(zz, tt_, mo_, ce_):
                    return denu(zz, tt_, encoder_hidden_states=[ce_, mo_],
                                pose_cond_fea=None, return_dict=False)[0].float()

                for i in range(n):
                    tt = torch.full((1,), float(sig[i]) * 1000.0, device=dev, dtype=dt)
                    v = torch.zeros_like(z, dtype=torch.float32)
                    cnt = torch.zeros((1, 1, F_, 1, 1), device=dev, dtype=torch.float32)
                    for _ix in _windows(self.val_ctx, F_, self.val_window, self.val_overlap, _rng):
                        s0 = torch.as_tensor(_ix, device=z.device)
                        zw = z[:, :, s0]
                        mw_c = mo_c[:, s0]; mw_u = mo_u[:, s0]
                        if w != 1.0:
                            self._set_ref(refu, rr, rw, it, ce, dev, dt)
                        vw = _pred(zw, tt, mw_c, ce)
                        if w != 1.0:
                            if rr is not None:
                                rr.clear()
                            vw_u = _pred(zw, tt, mw_u, ce_u)
                            vw = vw_u + w * (vw - vw_u)
                        v[:, :, s0] += vw          # 窗口内帧互不重复,索引累加安全
                        cnt[:, :, s0] += 1.0
                    v = v / cnt.clamp_min(1.0)
                    z = (z.float() + (sig[i + 1] - sig[i]) * v).to(dt)
            x = rearrange(z, "b c f h w -> (b f) c h w") / 0.18215
            dec = torch.cat([self._vae.decode(x[i:i + 8], x[i:i + 8].shape[0]).sample
                             for i in range(0, x.shape[0], 8)], 0)
            img = ((dec / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(np.uint8)
            gfs = sorted(os.listdir(it["gt_dir"])); GT = []
            for j, ti in enumerate(it["idx"]):
                gt = np.array(Image.open(os.path.join(it["gt_dir"], gfs[ti])).convert("RGB").resize((512, 512)))
                mse = float(((img[j].astype(np.float64) - gt.astype(np.float64)) ** 2).mean())
                P.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
                S.append(ssim_fn(gt, img[j], channel_axis=2, data_range=255))
                a = torch.from_numpy(img[j]).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                b = torch.from_numpy(gt).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                L.append(float(self._lpips(a, b)))
                nm = f"{it['clip'][:12]}_{int(ti):04d}.png"
                Image.fromarray(img[j]).save(os.path.join(gdir, nm))
                Image.fromarray(gt).save(os.path.join(rdir, nm))
                GT.append(gt)
            # ---- 可视化产物
            if self.kind == "image":
                # ★ GT 上 / 生成 下,横向排开**全部**验证帧。
                #   早先只存第一帧的左右并排图:8 clip × 1 帧信息量太小,
                #   肉眼看不出"哪个 ckpt 更好",而选 ckpt 恰恰只能靠肉眼。
                pair = np.concatenate(
                    [np.concatenate([GT[t], img[t]], 0) for t in range(len(img))], 1)
                Image.fromarray(pair).save(os.path.join(sdir, f"{it['clip'][:12]}.png"))
                sheet.append(pair)
            else:
                # ★ 视频类必须存 mp4:frame 0 是参考帧本身,所有 ckpt 都一样,存单帧毫无区分度
                self._write_mp4(np.concatenate([np.stack(GT), img], 2),      # 左GT右生成
                                os.path.join(sdir, f"{it['clip'][:12]}.mp4"))
                k = np.linspace(0, len(img) - 1, 6).astype(int)              # 帧条,快速扫
                Image.fromarray(np.concatenate(
                    [np.concatenate([GT[t], img[t]], 0) for t in k], 1)).save(
                    os.path.join(sdir, f"{it['clip'][:12]}_strip.png"))
            VG.append(np.stack(img)); VR.append(np.stack(GT))
        if sheet:
            # 各 clip 的验证帧数可能不同,按最窄的裁齐再纵向堆叠
            w = min(a.shape[1] for a in sheet)
            Image.fromarray(np.concatenate([a[:, :w] for a in sheet], 0)).save(
                os.path.join(sdir, f"_ALL_step{step:06d}.png"))
        _dd = self._dist()
        if _dd:
            # ★ 逐帧指标用 all_gather_object(小);整段帧走共享磁盘(每 clip 约 100MB,
            #   走 NCCL 会在已近满的显存上再占几百 MB)。rank0 按全局序号 gi 排序,结果与单卡逐位一致。
            import torch.distributed as _d
            fdir = os.path.join(sdir, "_frames"); os.makedirs(fdir, exist_ok=True)
            gis = [it["gi"] for it in items]
            for gi, a, b in zip(gis, VG, VR):
                np.save(os.path.join(fdir, f"g{gi:03d}.npy"), a); np.save(os.path.join(fdir, f"r{gi:03d}.npy"), b)
            _all = [None] * _dd[1]
            _d.all_gather_object(_all, (P, S, L))
            _d.barrier()                                   # 所有 rank 的帧与 png 都已落盘
            if _dd[0] != 0:
                return {}
            P = [x for part in _all for x in part[0]]
            S = [x for part in _all for x in part[1]]
            L = [x for part in _all for x in part[2]]
            gs = sorted(int(f[1:4]) for f in os.listdir(fdir) if f.startswith("g"))
            VG = [np.load(os.path.join(fdir, f"g{g:03d}.npy")) for g in gs]
            VR = [np.load(os.path.join(fdir, f"r{g:03d}.npy")) for g in gs]
            import shutil as _sh; _sh.rmtree(fdir, ignore_errors=True)
        out = dict(PSNR=float(np.mean(P)), SSIM=float(np.mean(S)), LPIPS=float(np.mean(L)))
        if self.kind != "image":
            out.update(self._temporal_metrics(VG, VR, dev))
        try:
            print(f"[val:prog] step {step} 算 FID(2048维协方差 sqrtm,单次约 76s)", flush=True)
            from pytorch_fid import fid_score
            out["FID"] = float(fid_score.calculate_fid_given_paths([gdir, rdir], batch_size=32,
                                                                   device=str(dev), dims=2048))
        except Exception as e:
            out["FID"] = None; print(f"[val] FID 跳过: {type(e).__name__}: {e}", flush=True)
        import shutil; shutil.rmtree(gdir, ignore_errors=True); shutil.rmtree(rdir, ignore_errors=True)
        return out

    # ---------------------------------------------------------------- 因果 rollout
    @torch.no_grad()
    def _causal_rollout(self, denu, ce, it, dev, dt):
        """逐 block 少步去噪 + clean-commit,步间用 **renoise**(与 src/distill/flow_rollout.py 一致)。

        ⚠️ 切勿改成 Euler:训练是 x̂0+新鲜噪声,推理用 Euler 等于用模型自己预测的噪声重加噪,
           分布不一致,实测 FVD 会翻一倍(文档 §2.8)。
        """
        sl = list(self.student_sigmas) + [0.0]
        F_ = it["x0"].shape[2]; blk = self.block
        g = torch.Generator(device=dev); g.manual_seed(self.seed)
        self.ctrl.set_mode("stream"); self.ctrl.reset_cache()
        outs = []
        for st in range(0, F_, blk):
            nb = min(blk, F_ - st)
            mo_b = it["mo"][:, st:st + nb]
            z = torch.randn(1, 4, nb, 64, 64, generator=g, device=dev, dtype=dt)
            for i in range(len(sl) - 1):
                self.ctrl.set_offset(st); self.ctrl.set_commit(False)
                v = denu(z, torch.full((1, nb), sl[i] * 1000.0, device=dev, dtype=dt),
                         encoder_hidden_states=[ce, mo_b], pose_cond_fea=None, return_dict=False)[0]
                x0p = z.float() - sl[i] * v.float()
                z = ((1 - sl[i + 1]) * x0p + sl[i + 1] *
                     torch.randn(x0p.shape, generator=g, device=dev, dtype=torch.float32)).to(dt)
            self.ctrl.set_offset(st); self.ctrl.set_commit(True)
            denu(z, torch.zeros((1, nb), device=dev, dtype=dt),
                 encoder_hidden_states=[ce, mo_b], pose_cond_fea=None, return_dict=False)
            outs.append(z)
        self.ctrl.set_mode("train"); self.ctrl.reset_cache()
        return torch.cat(outs, dim=2)

    # ---------------------------------------------------------------- 视频类专属
    @staticmethod
    def _write_mp4(arr, path, fps=25):
        import subprocess, tempfile
        h, w = arr.shape[1], arr.shape[2]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                             stdin=subprocess.PIPE)
        p.stdin.write(arr.astype(np.uint8).tobytes()); p.stdin.close(); p.wait()

    @torch.no_grad()
    def _temporal_metrics(self, VG, VR, dev):
        """FVD / DYN / HFE / TXF / FLK —— 口径与 decoder_bench 一致(§9.2/9.4/9.5)。
        ⚠️ 样本量远小于主表(8 clip vs 30),FVD 有强正偏,**只能与自己跨 step 比,不能对主表**。"""
        import torch.nn.functional as _F
        out = {}
        try:    # FVD:styleganv i3d,每 clip 前 16 帧,与 §9.2 一致
            import sys as _s
            _p = third_party("fvd")
            if _p not in _s.path: _s.path.insert(0, _p)
            from calculate_fvd import calculate_fvd
            n = min(16, min(len(v) for v in VG))
            def _t(V, size=224):
                x = torch.from_numpy(np.stack([v[:n] for v in V])).float().permute(0, 1, 4, 2, 3) / 255
                return _F.interpolate(x.flatten(0, 1), (size, size), mode="bilinear",
                                      align_corners=False).view(len(V), n, 3, size, size)
            r = calculate_fvd(_t(VG), _t(VR), device=str(dev), method="styleganv")["value"]
            out["FVD"] = float(list(r.values())[-1] if isinstance(r, dict) else r[-1])
        except Exception as e:
            print(f"[val] FVD 跳过: {type(e).__name__}: {e}", flush=True)
        try:    # DYN:RAFT 光流幅度比(§9.4)。★ 不能用帧差,帧差把闪烁当运动
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            if getattr(self, "_raft", None) is None:
                self._raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
            def mag(a, size=256):
                v = []
                x = torch.from_numpy(a).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1
                x = _F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
                for t in range(0, min(len(x) - 1, 32), 2):
                    fl = self._raft(x[t:t+1], x[t+1:t+2])[-1]
                    v.append(float(torch.sqrt((fl ** 2).sum(1)).mean()))
                return float(np.mean(v)) if v else 0.0
            g, r = np.mean([mag(a) for a in VG]), np.mean([mag(a) for a in VR])
            out["DYN"] = float(g / r) if r else None
        except Exception as e:
            print(f"[val] DYN 跳过: {type(e).__name__}: {e}", flush=True)
        # HFE / TXF / FLK(§9.5):三者必须一起看,单看 TXF 会被"糊"骗
        def hf(a, size=256):
            x = torch.from_numpy(a.astype(np.float32).mean(3)).unsqueeze(1).to(dev)
            x = _F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
            h = x - _F.avg_pool2d(x, 5, stride=1, padding=2)
            return float(h.abs().mean()), float((h[1:] - h[:-1]).abs().mean())
        hg, tg = zip(*[hf(a) for a in VG]); hr, tr = zip(*[hf(a) for a in VR])
        hfe, txf = np.mean(hg) / np.mean(hr), np.mean(tg) / np.mean(tr)
        out.update(HFE=float(hfe), TXF=float(txf), FLK=float(txf / hfe))
        torch.cuda.empty_cache()
        return out

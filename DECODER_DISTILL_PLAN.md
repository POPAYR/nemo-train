# XNeMo 解码器实时自回归蒸馏：训练计划与评估

> **任务**：把基于 Stable Diffusion 的 X-Nemo 视频解码器（25–35 步双向 DDIM、24 帧滑窗）
> 蒸馏成 **实时（25fps）、自回归/流式、少步（1–4 步）** 的因果解码器，方法 = **Self-Forcing / DMD**。
> 上游 motion AR transformer 已是因果/流式/少步（见 `motar/AR_MOTION_WORK_SUMMARY.md`），
> 本计划只针对**下游解码器**：`motion latent → 像素` 这一段。
>
> 文档日期：2026-06-25。基于对 `x-nemo-inference/src/`（解码器）、`Self-Forcing/`（方法）、
> `motar/`（上游 AR + 既有 self-forcing/GAN 经验）三处代码的通读。
>
> **已定约束**：① 部署目标 = 单卡 A100 @ 25fps（发论文，优先 25fps；有压力可降帧 + 插帧补齐）；
> ② 先走**纯 DMD**，细节/嘴部不足再按需加 GAN；③ 本文为正式计划文档。

---

## 1. 任务定义

X-Nemo 解码器当前在**两个轴**上都不实时，要同时消掉：

| 轴 | 现状（teacher） | 目标（student） |
|---|---|---|
| **步数** | 25–35 步 DDIM，ε-pred，CFG 2.5–3.5（每步 2× batch） | **1 步**（主配置，Phase 0 实测定死）；2 步 fallback；4 步仅离线质量参考 |
| **时序结构** | `temporal_modules` 双向 self-attn（max_len 32）+ 24 帧滑窗 + overlap 平均 | **block-causal + KV-cache 流式** |
| **条件（不变）** | ref-image（reference UNet，**整片缓存一次**）+ CLIP(768) + 逐帧 motion[T,32,16]（cross-attn dim16，**已逐帧因果**） | 原样保留 |

**实时预算（A100，已实测 → 详见 §5 Phase 0 结果）**：reference UNet 整片只跑一次（bank 缓存）≈ 摊销为 0；
student 推理丢 CFG（CFG=1.91× 成本）→ 逐帧成本 ≈ 去噪 UNet × 步数。
**实测**：1 步去噪 UNet（block K=8）= 20.7ms/帧 = **48fps，不是瓶颈**；瓶颈是 **SVD VAE decoder（70ms/帧）**，
换 **TAEHV 轻量 decoder** 后 1 步可达 ~33-42fps。→ **定死 1 步 + K=8 + TAEHV**，稳过 25fps；
这也与 motion 侧 §9.13「1 步推理更平滑」一致。

---

## 2. ⭐ 关键判断：解码器适合 DMD（与 motion 侧 §9.10 结论相反）

motar 在 motion AR 上得出「**DMD 不可用、只能 GAN**」——理由：序列层面**没有干净的双向 teacher**，
唯一的因果 AR teacher 被 exposure bias 污染，real-score 不可信（详见 `AR_MOTION_WORK_SUMMARY.md §9.10`）。

**对解码器，这个死结不存在**：

- 解码器的 teacher = **冻结的 X-Nemo 多步双向 UNet**，它本身是干净的双向扩散模型
  （非自回归、无 exposure bias），**正是 DMD 需要的 real-score**。
- 因此解码器蒸馏是**教科书式的 Self-Forcing/DMD 场景**，GAN 只是可选润色而非唯一解。
- 这把任务风险从「motion 侧那种脆弱 GAN 调参」降到「标准 DMD + 一次架构手术」。

> **一句话**：你们在 motion 侧绕不开 GAN，是因为缺干净 teacher；解码器自带干净 teacher，所以可以放心用 DMD。

---

## 3. 系统总览与组件映射

```
 audio+text ─► [因果 motion AR](已完成) ─► motion latent [T,512]
                                                  │
        ref image ─► CLIP emb + VAE ──┐           │
                                       ▼           ▼
                         ┌──────────────────────────────────────┐
                         │  因果少步 X-Nemo 解码器 (student)       │  ← 本计划产物
                         │  temporal_modules: block-causal+KVcache │
                         │  motion/spatial/ref: 原样(已因果/已缓存) │
                         └──────────────────────────────────────┘
                                       │  1–2 步去噪
                                       ▼
                            (TAEHV/SVD) VAE decode ─► 视频帧 (25fps 流式)
```

| Self-Forcing 组件 | X-Nemo 对应 | 工作量 |
|---|---|---|
| `real_score`（冻结双向 teacher） | 冻结 X-Nemo 去噪 UNet（25 步 DDIM + CFG） | 复用，0 |
| `generator`（因果少步 student） | **因果版** X-Nemo UNet（temporal 走 `bank`/KV-cache） | **主要新工作** |
| `fake_score`（在线 critic） | X-Nemo UNet 副本，ε-去噪 loss；建议 **LoRA-critic** 省显存 | 中 |
| flow↔x0 换算 | 改成 **DDPM ε↔x0**（motar 已有 v-pred/DDIM/x0 反演代码） | 小 |
| block-causal mask + 偏移 PE | 只改 `temporal_modules`（spatial/motion/ref 不动） | 中 |
| 文本条件 | ref-bank（缓存）+ 逐帧 motion token | 复用 |
| ODE-init（CausVid 回归） | 真实 (motion, ref) 上回归 teacher 的 DDIM 轨迹 | 中 |
| TextDataset（data-free） | `motar/data/dataset.py: MotarDataset(load_video=True)` 已就绪 | 复用 |

**X-Nemo 比 Wan 好改的两点**：
1. **时序混合被隔离**在独立的 `temporal_modules`（`Temporal_Self`）里，spatial / motion / reference 完全不用动。
2. **每帧已有强 motion 条件**（逐帧 32×16 token cross-attn），temporal attn 主要管一致性/平滑而非生成内容
   → **双向→因果的信息损失天然比纯 T2V 小**，ODE-init 起点更好，self-forcing 主要做收尾。

---

## 4. X-Nemo 解码器架构事实（与蒸馏相关，带代码锚点）

- **去噪 UNet**：SD-1.5 inflate 成 3D；每个 block 串 `attentions`（spatial self+cross→CLIP）、
  `motion_modules`（`VanillaTemporalModule`，`[Spatial_Cross]`，**cross_attention_dim=16**，逐帧 motion 条件）、
  `temporal_modules`（`[Temporal_Self]`，**双向**，PE max_len=32）。
  `unet_3d_blocks.py:301,331`（两类模块并列迭代）。
- **非因果根因 = `temporal_modules`**：`motion_module.py:438-483` 把 `(b f) d c -> (b d) f c` 后对**全窗 f 无掩码** self-attn，
  绝对正弦 PE。逐帧 motion 条件与 reference 都已因果，**唯一要改的就是这个**。
- **⭐ 因果 hook 已预留**：`motion_module.py:446-459` —— 当传入 `bank` 时，temporal self-attn
  自动变成对 `[bank(过去帧)] + [当前帧]` 的 **cross-attn** 并重施 PE。
  注释原文「motion_frames作为之前的帧」。**全仓没人填 `bank`（死代码）**，但正是 KV-cache 因果化的现成接缝。
- **reference attention 已缓存**：reference UNet 整片只跑一次写 bank，逐帧广播、step 不变
  （`mutual_self_attention.py:354-368`）→ **天然流式兼容**，逐帧成本摊销为 0。
- **扩散参数化 = ε-pred DDPM**（`configs/inference/inference_xnemo_stage2.yaml:44`），DDIM 推理、CFG（motion 用 `neg_motion` 中性 token + CLIP 置零）。
- **VAE**：SVD `AutoencoderKLTemporalDecoder`，8× 空间下采样、**无时序压缩**（1 latent 帧 = 1 视频帧），scaling 0.18215。
- **参数量**：去噪 UNet ≈1.69B（base 859M + motion 372M + temporal 454M）+ reference 860M + motion-encoder 62M。
- **数据已就绪**：`motar/data/dataset.py` 在 `load_video=True` 时返回
  `motion_tensor[T,512] / video_tensor(VAE latent)[T,4,64,64] / ref_latent / ref_img`，
  即 ODE-pair 生成、DMD 条件、（将来）GAN real 样本全部齐备。数据集 `MEAD_frames_512_25fps`（25fps）。

---

## 5. 分阶段计划

### Phase 0 — 实时可行性闸门（先做，1–2 天，可一票否决）
**目的**：在动手前定死步数预算，避免「蒸完发现达不到 25fps」。
- benchmark 单次去噪 UNet forward @64×64 在 A100 上的延迟（batch=1，逐 block 帧数 K∈{1,2,4}）。
- 推算 1/2/4 步 × VAE decode 的端到端 fps；reference UNet 单独计时（确认可摊销）。
- VAE decode 也要单独量：SVD temporal decoder 可能是瓶颈 → 预案 **TAEHV 轻量 decoder**（Self-Forcing 自带思路）。
- **产出**：一张「步数 × K × fps」表 + 决策：
  - 若 1 步可达 25fps → 主配置 = **1 步**（DMD2 能蒸到 1 步）。
  - 若需 ≤2 步 + TAEHV/compile/CUDA-Graph → 记入工程清单。
  - 若 4 步才够质量但超时 → 文档化「4 步质量配置 + 降帧 + 插帧」作为论文 fallback。
- **额外**：torch.compile + 静态 KV-cache + CUDA Graph 的收益评估（motar §9.4 已知 kernel 启动是瓶颈）。

#### ✅ Phase 0 实测结果（2026-06-25，单 A100 80GB PCIe，fp16 eager，脚本 `scripts/val/bench_decoder_phase0.py`）

| F=block 帧 | 去噪 UNet (batch=1, 无CFG) | ms/帧 | SVD VAE decode ms/帧 |
|---|---|---|---|
| 1 | 78.7 ms | 78.7 | ~63 |
| 2 | 87.5 ms | 43.7 | ~72 |
| 4 | 102 ms | 25.5 | ~70 |
| **8** | **165 ms** | **20.7** | ~70 |
| 16 | 306 ms | 19.2 | ~71 |
| 24 | 458 ms | 19.1 | ~72 |

去噪 UNet=1684.8M，reference UNet=859.5M（整片一次 31ms，摊销≈0），SVD VAE=97.7M。
固定 per-forward 开销 ~60ms → **block 必须 ≥8 帧才把它摊掉**（F=1 是灾难性的 78ms/帧）。
CFG（batch=2）= **1.91×** 成本 → student 推理必须丢 CFG（DMD 蒸出的 student 本就无 CFG）。
功耗仅 ~70W/250W → UNet 是 **launch/latency-bound 非 compute-bound**（compile/CUDA-Graph 有空间）。

**fps（生成 K 帧 = S·t_unet(K) + t_vae(K)）：**

| | S=1 (unet \| +SVD-vae) | S=2 | S=4 |
|---|---|---|---|
| K=4 | 39.2 \| 10.5 | 19.6 \| 8.3 | 9.8 \| 5.8 |
| **K=8** | **48.4 \| 11.0** | 24.2 \| 9.0 | 12.1 \| 6.5 |

**⭐ Phase 0 结论（闸门通过，但有一个强制项）：**
1. **1 步去噪 UNet 不是瓶颈**：K=8 时 48fps，远超 25fps；2 步 24fps 临界；4 步 12fps 出局。
2. **SVD VAE decoder 是唯一瓶颈**（~70ms/帧 → 全管线封顶 ~11fps），且 decode > UNet
   → **流水并行也救不了，必须换轻量 decoder**（Self-Forcing 正是这么做）。这是 25fps 的**强制前提**。
   - ⭐ **XNeMo latent = 4 通道 SD latent（scaling 0.18215）→ 直接兼容的是 `TAESD`（madebyollin，4ch SD），不是 TAEHV（16ch Wan/Hunyuan）**。
     TAESD decode ~1-2ms/帧，是 drop-in（同 latent 空间）。若丢失 SVD 的时序平滑，可在 XNeMo latent→GT 帧上微调 TAESD（或自蒸一个时序版）。
3. **换 TAESD 后估算**（保守按 ~10ms/帧）：K=8/S=1 → 165+80=245ms/8 ≈ **33fps**；TAESD 实测 ~1-2ms/帧 → **~45fps**。
   → **1 步 + TAEHV-VAE 在单 A100 上稳过 25fps，且留有 KV-cache temporal 开销的余量。**
4. **步数预算定死 = 1 步**（目标）；2 步作 fallback（~22fps，需 compile/插帧补）；4 步仅作离线质量参考。
5. **block K=8**（causal `num_frame_per_block=8`，chunk 延迟 320ms@25fps，可接受）。
6. 备用余量杠杆（非必需）：torch.compile 实测 167→140ms（1.19×）；temporal/motion 模块当前用 vanilla
   `AttnProcessor`（非 SDPA）→ 可换 `AttnProcessor2_0` 提速；channels_last / FP8。

> **一句话**：可行性闸门**通过**。瓶颈不在扩散步数而在 VAE decoder；**1 步蒸馏 + TAESD VAE = 单 A100 实时 25fps**。

#### ✅ TAESD 验证 + 部署态显存（2026-06-25，脚本 `scripts/val/bench_mem_taesd.py`）

- **TAESD 画质 PASS**：真实人脸帧 latent，TAESD(2.45M) decode **PSNR 32.3dB vs 原图**（SVD 上界 33.8dB，**仅差 1.57dB**；
  TAESD vs SVD 输出 32.9dB 近乎一致）。并排图 `output/taesd_check/orig_svd_taesd.png`。正确缩放 = `z×0.18215`（UNet 空间 latent）。
- **TAESD 速度 PASS**：**2.83 ms/帧 vs SVD 69.85 ms/帧 = 25× 加速**。
- **⭐ SVD decoder 还是个显存炸弹**：K=8 一次解 8 帧 512px 峰值 **58.5 GB**！TAESD 逐帧、峰值仅 7.5GB。→ TAESD 不只是提速，更是 24G 可行的前提。
- **部署态推理显存（fp16，1 步，K=8，TAESD，无 SVD）**：权重常驻 **5.37 GB**，全流程峰值 **7.5 GB**。
- **实时 fps（A100 实测合成）**：UNet 165ms + TAESD 23ms = 188ms/8 帧 = **42.6 fps**（VAE 流水并行则 UNet-bound 48fps）。

**部署/扩展速查（回答常见硬件问题）：**

| 问题 | 结论 |
|---|---|
| 实时配置 | **1 步去噪 + block K=8 + 丢 CFG + TAESD decoder** → A100 ~42fps |
| 24G 显卡能推理吗 | **能，轻松**（峰值 7.5GB；Self-Forcing 本身就跑 RTX 4090/24G）。注意：这是**推理**；DMD **训练**显存大得多（3 模型+梯度+rollout），需多卡 |
| H100 预计加速 | **~1.5–2×（开箱）**：工作负载是 launch/latency-bound（A100 仅吃 70W/250W），裸 FLOPS 提升不直接兑现（呼应 motar §9.4 实测 H100 单流仅 ~1.3×）；配 torch.compile+CUDA-Graph 杀掉 launch 开销后可逼近 ~2.5–3×。**25fps 不需要 H100** |
| FP8 量化是否提升 | **A100(Ampere) 无 FP8 张量核 → 不提速**（仅省显存）。**H100/4090(Hopper/Ada) FP8 可 ~1.5–2× + 权重减半**，但 batch=1 需配 CUDA-Graph 才显效，且 **1 步模型质量裕度薄、FP8 有掉质风险**（须 PSNR/FVD/SyncNet 实测）。**达 25fps 不需要 FP8** |

### Phase 1 — 因果架构手术（核心工程，1–2 周）
**只改 `temporal_modules`**，spatial / motion_modules / reference 全部不动。
1. **训练态 block-causal mask**：temporal self-attn 加 block-causal 掩码（块内双向、跨块只看过去），
   照搬 Self-Forcing `wan/modules/causal_model.py:_prepare_blockwise_causal_attn_mask` 的 flex_attention 思路。
   绝对正弦 PE → 带 `start_frame` 偏移（对齐 KV-cache 的绝对位置）。
2. **推理态 KV-cache**：落地 `motion_module.py` 的 `bank` 路径——
   每个 temporal module / 每个空间位置 d / 每个分辨率，把过去帧 feature 入 cache；
   self-attn 变 cross-attn over `[bank]+[current]`；配 **local-window + sink 帧逐出**（无限流式、显存有界，
   参照 Self-Forcing `CausalWanSelfAttention` 的 roll + sink）。
3. **block 粒度**：`num_frame_per_block = K`（1 latent 帧 = 1 视频帧 = 40ms）。K 越小延迟越低、漂移越大；
   建议从 **K=4** 起调（160ms chunk），Phase 0 的 fps 结果会进一步约束 K。
4. **验收（硬指标）**：teacher-forcing 下，因果 student 与原双向 UNet 做**逐元素对拍**
   （类比 motar 的 KV-cache vs 整窗前向对拍）。数值一致才算 Phase 1 通过。
- **文件改动**：`src/models/motion_module.py`（causal mask + KV-cache）、
  `src/models/unet_3d.py` / `unet_3d_blocks.py`（透传 cache、block 配置）、
  新建 `src/pipelines/causal_streaming_pipeline.py`（流式推理）。

#### ✅ Phase 1 进度（2026-06-25）—— 核心架构 + 对拍闸门 PASS

**已实现**（`src/models/motion_module.py` + 新增 `src/models/temporal_causal.py`）：
- `VersatileAttention._causal_temporal_attn`：时序自注意力的因果版（直接 to_q/k/v/out + SDPA），两模式：
  - `train`：整窗前向 + **block-causal mask**（`build_block_causal_mask`，块内双向、跨块只看过去，可叠 local window）。
  - `stream`：**KV-cache** 逐 block，当前 block 的帧 attend `[cache + 当前]`，可 window 逐出旧帧（有界显存）。
- `PositionalEncoding.forward(x, offset)`：**绝对位置编码按 block 偏移对齐**（train/stream 一致的前提）。
- `TemporalCausalControl`：模块级状态注入（仿 ReferenceAttentionControl），`set_mode/set_offset/set_block_size/set_window/reset_cache`；
  **默认 `_causal_mode='off'` → teacher 与现有 pipeline 零影响**。只作用于 42 个 Temporal_Self 模块；motion/spatial/reference 不动。
- 验收脚本 `scripts/val/test_causal_parity.py`（3 级对拍，全 PASS）：
  - **P1** causal-train(block=F) == 原始双向：**逐元素精确 Δ=0**。
  - **P2** 整 UNet end-to-end stream(KV-cache) == causal-train(mask)：rel **6.7e-5**（fp32 经 42 模块+深 UNet 放大，benign）。
  - **P3** fp64 孤立单元 train==stream：**1.55e-15（机器精度）**，full-history 与 local-window 均精确 → 证明 cache/mask/PE 逻辑无误。
  - 健全性：block-causal vs 双向 Δ=0.57（证明因果确实裁掉未来，不是 no-op）。

**Phase 1 续（2026-06-25 后续）**：
1. ✅ **PE 自动扩展 >32 帧**：`PositionalEncoding` 改 `persistent=False` + 按需重算正弦（位置 0..31 不变），
   流式 >32 帧不再越界；配 local-window 逐出（有界显存）。**对拍补 F=40/window=24 case，仍机器精度 1.55e-15 PASS。**
2. ✅ **`causal_streaming_pipeline.py`**（`CausalStreamingPipeline.stream`）：因果 UNet + reference bank(缓存一次)
   + 逐帧 motion + **TAESD** 串成流式管线；逐 block N 步 DDIM（commit=False 只读干净历史）+ block 干净后 commit 写缓存；
   加 `_causal_commit` 开关（多步去噪中间步不污染缓存）。
3. ✅ **真实样本可视验收**（`scripts/val/test_causal_stream_video.py`，MEAD M003，同 ref/motion/init_noise）：
   - **管线端到端跑通出视频**；causal(block=8) vs bidir(block=T) 同噪声对比，像素 mean|Δ|=0.115。
   - **⭐ 关键观察**：**steady-state（KV-cache 填满后，如 frame24/block3）causal ≈ bidir，干净人脸几乎一致**；
     伪影集中在 **cold-start block（frame0-7 只有 8 帧上下文 → 对 24-32 帧训练的模型 OOD）**。
   - **结论**：因果机器正确（对拍已证）；未蒸馏的 causal 在 cache 填满后已可用，cold-start 伪影正是 **ODE-init/DMD 要修的**
     （蒸馏会在因果/短上下文 regime 训练 temporal 模块）。这也实证了"必须蒸馏、不能直接因果化"的动机。

**Phase 1 仍待（非阻塞，可并入 Phase 2/3）**：
- ⬜ sink tokens（保留首帧）配合 window 逐出，长流式更稳。
- ⬜ cold-start 处理（首 block 用 ref-latent prefill / 或交给蒸馏训练覆盖）。
- ⬜ `num_frame_per_block=8` 定为部署默认（已用于验收）。
- ⬜ commit 前向的开销优化（1 步部署时 1 去噪+1 commit=2× forward；Phase 3 再定是否复用末步 K/V）。

### Phase 2 — ODE 初始化（CausVid 式，几天 + GPU）
Self-Forcing 中 ODE-init 实质必需（稳定蒸馏前提，`strict=True` 加载）。
1. **造 ODE pairs**：冻结 teacher 在真实 `(motion latent, ref image, 真噪声)` 上跑 DDIM-N 步（如 N=25），
   存子采样轨迹 `[纯噪 → 3 中间 → x0]`（**用真实数据做条件，但目标是 teacher 自己的解，不需要 GT 像素**）。
   脚本仿 `Self-Forcing/scripts/generate_ode_pairs.py`。
2. **回归**：因果少步 student 单步预测某轨迹点 x0，对 teacher 解做 MSE 回归
   （仿 `model/ode_regression.py`）。得到**因果 + 少步**但 train/test 失配的起点。
3. 此阶段同时吃掉「降步 + 加因果」两件事；剩余因果 gap 交给 Phase 3。
- **产物**：`ode_init_causal.pth`（因果少步 student 初值）。

#### 🔄 Phase 2 进度（2026-06-25，训练中）

**⭐ 设计修正（实测驱动）**：原计划"ODE-init 一步到位降步+加因果"。实测发现 **eps→x0 的 1-step 回归在 fp16 高 t 数值爆炸(NaN)**，
且"回归到单个 1-step x0"本质就是 **DMD 要修的均值模糊**（回归=条件均值）。→ **拆分**：
- **ODE-init 只做"因果适配"**：eps-MSE（teacher 原生目标，bf16 稳定）+ block-causal(block=8)，把双向 temporal 模块稳稳改成因果、
  修 cold-start，**保持多步去噪能力**。不在这里强行降步。
- **降步到 1-step 交给 Phase 3 DMD2**（分布匹配，专治 1-step 均值模糊）。若 DMD 直接从多步起步不稳，再插一个 LCM/consistency 中间步。

**已实现 + 启动**（`scripts/train/ode_init_decoder.py`）：
- **数据：直接复用 `train_ar_xnemo.py` 的加载**——`MotarDataset(load_video=True)` + `ConcatDataset`，
  源(MEAD+hallo3)与参数读 `configs/train_ar.yaml` 的 data 段（用 `data_name_path` 索引，不扫盘）。**MEAD+hallo3 共 210522 clip**。
  （不重造 dataset，避免与其他脚本不一致。）batch 关键字 motion_tensor(归一化→denorm 喂 UNet)/video_tensor(=x0)/ref_latent/ref_img(→CLIP)/mask。
  **无需 teacher 采样**（video_tensor 即真实 x0）。实测确认缩放：video_tensor/ref_latent std≈1.06（已是 UNet 空间，直接用，不再 ×0.18215）、denorm 后 motion std≈1.0。
- **⭐ 可训练范围 = 只训 `temporal_modules`（~454M）**，spatial / motion_modules / reference UNet / CLIP 全冻结。
  理由：bidirectional→causal 的**结构改动只在 temporal 模块**，cold-start 也是纯 temporal 问题；冻结 spatial/motion 直接**保留 teacher 的外观/身份/口型质量**（零遗忘），更轻更快更稳。（`--train_scope temporal`；`all` 选项保留备用。）
- loss=eps-MSE, t~U[20,980], bf16, grad-checkpoint, L=16 帧, B=1×accum4, lr=1e-5, TemporalCausalControl(train 模式, block=8)。
- **后台训练已在 GPU5 启动**（log `output/ode_init/train.log`，keep_last=3 防爆盘，每 2000 步存）。
- 踩坑修复：① ref_latent 双 batch；② **fp16 x0 转换 NaN → 改 bf16 + eps-MSE**；③ add_noise 升 fp32 → 回 bf16；④ reference bank dtype 对齐(bf16)；⑤ **ckpt 每个 3.2GB + 盘 100% 满 → keep_last=3 封顶**；⑥ pkill -f 误杀自身 shell。

**✅ step-2000 渲染验证（2026-06-26，`scripts/val/render_ode_ckpt.py`，teacher-causal vs ODE-init-causal 同 ref/motion/noise）**：
- **cold-start（frame 4）：teacher-causal 彩噪 → ODE-init 干净人脸**（仅 2000 步 ~4.5h，temporal-only 453M）。**cold-start 伪影基本消除。**
- warm（frame 24）：两者都干净、可比 → **ODE-init 修好 cold-start 且不损 steady-state**。
- eps-MSE 几乎平（~0.037）= 预期：从已很强的 teacher 起步，causal 适配只是小 delta，数值不动；真正变化是**定性的 cold-start 修复**（只有渲染看得出）。
- 产物 `output/ode_render/..._frame{4,24}_teacher_vs_odeinit.png` + `..._teacher_vs_odeinit.mp4`。**Phase1(因果架构)→Phase2(ODE-init) 端到端验证成功。**

**待办**：① ODE-init 继续多跑几千步求稳（周期性渲染确认），挑稳定 ckpt 作 Phase 3 起点；
② 定 Phase 3 起步方式（DMD2 直接降步 vs 先 LCM 中间步）。

### Phase 3 — Self-Forcing DMD 蒸馏（主菜，1–2 周调参）
1. **self-forcing rollout**（仿 `pipeline/self_forcing_training.py`）：block-wise AR，
   每块跑少步采样但**只在随机 1 个 exit step 开梯度**、**梯度只回流最后 N 帧**、**用 clean prediction 写 KV-cache**。
   这三件套是 train==test 的关键，也是让长 rollout 显存有界的机制。
2. **DMD generator loss**（ε/DDPM 空间，见 §6）：`grad ∝ (x0_fake − x0_real_cfg)`，
   `L_G = 0.5·MSE(x0, sg(x0 − grad))`；teacher 带 CFG（motion=neutral + CLIP=0，guidance≈2.5）。
3. **critic 在线更新**：ε-MSE 去噪 loss 追 student 当前分布；**gen:critic = 1:5**。
4. **rollout 要够长**：motar 学到 drift 要 ~128 帧才显形；解码器 exposure bias = 外观/误差累积，
   rollout 也要够长（用 last-N-frame 梯度门控控显存）。建议 rollout ≥ 64 帧、N(梯度窗) 取 21–32。
5. **条件分布**：主用**真实 motion latent**（让解码器当忠实渲染器），可少量混入 AR 生成 motion 增鲁棒性，
   **但不把解码器质量耦合进 AR 质量**（解码器只负责「忠实渲染给定 motion」）。
- **⭐ 可训练范围（与 ODE-init 不同——这里要降步，须动去噪行为，不止 temporal）**：
  - **Generator（因果 student）**：`temporal_modules` 继续全量训（因果），spatial/motion 的 1-step 适配走 **LoRA**（attention+conv 低秩）。
    即 generator = teacher 基座(冻) + temporal(训) + spatial/motion LoRA(训)。比 ODE-init 多动 spatial，因为 1-step 生成质量由整条去噪路径决定。
  - **Critic（fake_score，双向）**：teacher 基座冻结 + **LoRA**（DMD2 标准做法，critic 只需当 student 分布的去噪器）。
  - **Teacher（real_score）**：全冻结。
  - **为何 LoRA**：3 份 ~1.69B UNet（gen+critic+teacher）若 gen/critic 全量微调，单卡放不下（光优化器态就 ~40GB）；
    LoRA 把可训练参数压到几十 M → 优化器/梯度极省，显存由 3 个冻结基座 + rollout 激活主导，1–2 卡可行（多卡 FSDP 更宽裕）。
- **显存**：3×1.69B 基座 + ref(0.86B 冻) + TAESD；FSDP + grad-checkpoint + last-N-frame 门控 + LoRA。
- **文件**：新建 `scripts/train/distill_decoder_dmd.py`、`src/distill/{dmd.py,rollout.py,critic.py}`、
  `configs/distill/decoder_dmd.yaml`。

#### 🔜 Phase 3 实现设计（2026-06-26，已读 Self-Forcing 参考实现，开始实现）

**起点**：generator init = `output/ode_init/ode_step_8000.pt`（ODE-init 验证通过：cold-start 已修、稳定）。

**3 个网络（都是 X-Nemo 去噪 UNet，ε-pred DDPM）**：
| 网络 | 结构 | 训练 | 备注 |
|---|---|---|---|
| generator（因果 student） | 因果版(TemporalCausalControl stream) + temporal 全训 + spatial/motion LoRA | ✅ | 从 ode_8000 起；few-step `denoising_step_list`（先 4 步 `[999,749,499,249]`，目标压到 1 步）|
| fake_score（critic，双向）| teacher 基座 + LoRA | ✅ | 追 student 当前分布的去噪器 |
| real_score（teacher，双向）| 原始冻结 UNet | ❄️ | 带 CFG（motion=neutral + clip=0，guidance≈2.5）|

**rollout**（仿 `pipeline/self_forcing_training.py`，用我已有的 TemporalCausalControl stream + `_causal_commit`）：
逐 block（block=8）跑 few-step；**每 block 只在随机 1 个 exit step 开梯度**、**梯度只回流最后 N 帧**、
**block 干净后用 commit 前向写干净 KV**。条件 = (motion, ref bank)。

**DMD2 loss（ε/DDPM 空间，见 §6）**：
- generator：`x0 = rollout输出`；采 t、加噪 `x_t`；`x0_fake = eps→x0(fake_score(x_t))`、`x0_real = eps→x0(teacher_cfg(x_t))`；
  `grad = (x0_fake − x0_real) / mean|x0 − x0_real|`；`L_G = 0.5·MSE(x0, sg(x0 − grad))`（gradient_mask 屏蔽首块/pad）。KL-grad 全程 no_grad，仅末 MSE 可微。
- critic：no_grad 跑 rollout→generated；加噪；fake_score 预 eps；**ε-MSE 去噪 loss**（X-Nemo 原生，比 flow 简单）。
- **gen:critic = 1:5**（critic 每步更，generator 每 5 步更）。

**关键适配（vs Self-Forcing 的 Wan/flow）**：① 用 ε→x0 的 DDPM 反演（motar 已有），不用 flow；② 每帧 1 latent=1 视频帧（无时序压缩），block=8；③ 条件走 reference bank(缓存一次)+逐帧 motion，不是 text；④ 全部用 LoRA 控显存（3×1.69B 基座）。

**实现顺序（增量 + smoke）**：① model 封装(3 网络统一 `(eps,x0)` 接口 + LoRA + ode_8000 加载) → ② ε-空间 DMD loss(dummy 测) → ③ rollout 适配(单 block grad 测) → ④ critic loop + 1:5 → ⑤ 数据复用 MotarDataset → ⑥ smoke → ⑦ 启动 + 监控(FVD/SyncNet/std)。
**文件**：`scripts/train/distill_decoder_dmd.py` + `src/distill/{models.py,rollout.py,dmd_loss.py}` + `configs/distill/decoder_dmd.yaml`。

#### 🔄 Phase 3 进度（2026-06-30，run `output/dmd2_native_0629/`，**忠实复刻 native Self-Forcing DMD**：全量微调非 LoRA、纯 DMD 无 anchor、带 EMA）

**⭐ 当前 run 关键参数（截至 step ~8900 健康：x0std≈1.0、G≈0.007、C≈0.05，21s/it）**：

| 项 | 值 | 备注（= native 对齐处） |
|---|---|---|
| 起点 gen_ckpt | `ode_init/ode_step_8000.pt` | ODE-init（因果适配完成）；本 run 实际 `--resume output/dmd2/dmd2_step_4000.pt`（同谱系 native ckpt 续跑） |
| 可训练范围 | **gen 全量 + critic 全量**（各 1684.8M，非 LoRA） | log `[opt] full-finetune`；native 即全量微调 |
| denoising_step_list | `[999, 749, 499, 249]`（**4 步**，train==infer） | native `[1000,750,500,250]`（DDPM 直接 timestep，不做 flow shift/warp） |
| block（num_frame_per_block） | **8**（1 latent=1 视频帧；无 overlap，块内双向+跨块因果） | native=3（Wan 有 4× 时序压缩，不可直接比） |
| window（KV-cache 历史） | 0 = 全历史（32 帧序列内不逐出） | 长流式再开局部 window+sink |
| gen_lr / critic_lr | **2e-6 / 4e-7** | native lr / lr_critic（曾因 50–100× 过高发散，已对齐） |
| betas / wd / grad_clip | (0, 0.999) / 0.01 / 10.0 | native 全对齐 |
| gen:critic 更新比 | **1:5**（critic 每步、gen 每 5 步） | native dfake_gen_update_ratio=5 |
| EMA | decay 0.99，start step 200，部署用 `generator_ema` | native ema_weight/ema_start；**部署务必用 EMA 权重** |
| CFG / guidance | **0（关闭）** | native=3.0，但需 neg_motion plumbing → 列为下一质量项 |
| 并行 / batch | 双卡 GPU5,6，B=3/卡 × accum2 × 2卡 = **有效 batch 12** | 手写 grad all-reduce DDP（非 module-wrap，避开 rollout 多次调用脆弱性） |
| L（序列帧） / grad_frames | 16 / 末 block（last-N-frame 梯度门控） | 显存有界 |
| loss 空间 | **ε-pred DDPM**：DMD2 梯度在 x0 空间、critic ε-MSE | 非 flow（flow shift/warp/flow-loss 是 Wan 专属，未移植） |
| 显存 / 存储 | 63.1GB/卡（优化器态 27GB 主导）；ckpt 10GB(gen+critic+ema)，keep_last=3 | ssd5 盘紧，注意清理 |
| 文件命名铁律 | 每 run 独立目录 + 时间戳 log，ckpt/eval 在该目录下 | 防 keep_last 误删他 run（曾踩坑） |

**纯 DMD、无 anchor**：之前 LoRA + 高 lr + 自加 anchor 的版本反复发散（step 6000/11000 runaway）；本 native run 把 lr 降到官方值 + 全量 + EMA + 去 anchor 后**稳定**（详见记忆 [[decoder-distill-project]]）。

**已知现象（block 边界跳变，2026-06-30）**：渲染每 8 帧(block 边界 frame8/16/24)可见跳变。根因 = 因果 block **不重叠**(替掉了 teacher 原来的 24 帧滑窗+overlap 平均)，块内双向、跨块只靠 commit 干净 KV 续接。**官方 Self-Forcing 同样无 overlap**，靠 self-forcing rollout 训练学边界续接 → 属预期、应随 DMD 收敛减弱；**不加回 overlap**（破坏流式/实时，且非官方做法）。可选官方旋钮：`context_noise`（给 cache 上下文加微噪，默认 0）、block 大小。

#### ⭐ 跳变根因诊断 + 长 rollout 修复（2026-07-01）—— rollout 太短，向官方对齐

**诊断（看 step-10000 渲染跳变明显）**：跳变**不是玄学，是 rollout 太短**。
- 旧 run 实测：`L=16`(=2 block) + `grad_frames=8`(**只有末 1 block 有梯度**) → 模型**只练过 1 个 block 边界**。
- 但渲染 32 帧 = 4 block = **3 个边界**(frame8/16/24)，boundary 2/3 **模型从没见过 = 纯 OOD → 跳变**。
- 对照官方：21 帧=**7 block**、`start_gradient_frame_index=num_output_frames−21=0` → **几乎全帧有梯度**；且本计划 §Phase3 自己写的就是「rollout ≥64 帧、梯度窗 21–32」。**我们为省显存把 self-forcing 最核心的长 rollout 阉割了。**

**修复（忠实官方，`scripts/train/distill_decoder_dmd.py` 已重写）**：
| 项 | 旧 | 新 | 依据 |
|---|---|---|---|
| `L`(rollout) | 16 | **64**(8 block=**7 边界**，≈官方 21帧/7block) | 拉长练边界；前置 block no_grad ≈ 免费 |
| `grad_frames` | 8(末1块) | **-1=全部帧** | 官方 `start_gradient_frame_index=0` 全帧梯度 |
| `batch`/卡 | 3 | **1** | 省显存换长 rollout（说话人脸方差小，小 batch 够） |
| `accum` | 2 | **4**（有效 batch=1×4×2卡=8） | 补偿 B=1；batch8 native 稳定区间 |
| 显存 | 手写 all-reduce(全量优化器 27GB/卡) | **`ZeroRedundancyOptimizer`(ZeRO-1 优化器态分片)** | 腾空间给全帧梯度 |
| lr/EMA/betas/wd/clip/ratio/dsl | native | **不动** | 只改 rollout/显存，不碰稳定性 |

- **ZeRO 选型说明（诚实记录）**：真·ZeRO-2(DeepSpeed/FSDP 包 generator)会 wrap 模型，而 rollout **一步内多次调用 generator**(部分 no_grad)——正是记忆里「FSDP/DDP-wrap 在多次调用 rollout 上太脆弱」而改手写 all-reduce 的原因。故用 PyTorch `ZeroRedundancyOptimizer`(**ZeRO-1：只分片优化器态=那 27GB 主开销，不分片参数** → EMA/存档/rollout 全不受影响，robust)。梯度分片(ZeRO-2 多出的部分)省的是较小的 ~7GB 且要冒 wrap 险，收益低。存档前 `consolidate_state_dict` 汇聚。
- **显存 smoke 实测（单卡最坏=未分片，`--grad_frames -1` 全帧梯度）**：L=32→55.4GB、L=48→68.3GB、L=64→82.3GB(单卡到边缘)。**墙钟 ≈ accum×L**（两者都顶满 18–24 天不现实）→ 定 L=64 + accum=4。
- ~~已启动 dmd2_longroll_0701 L=64 全帧梯度~~（**已停，被下方 teacher-OOD 修正取代**）：76GB/卡、~50s/it。

#### ⭐⭐ 关键修正 2（2026-07-01）—— teacher 只在 24 帧上训，全帧打分 = teacher OOD → 改「随机 24 帧窗打分」

**问题（用户发现）**：全帧梯度要把 rollout 的 **64 帧一次性喂 teacher(real_score)** 打分。但 **teacher 是原始 X-Nemo 双向 UNet，只在 24 帧滑窗上训过、PE max_len=32**。喂 64 帧 → **PE 位置 32–63 未训 + 双向注意力范围翻倍 = OOD**，`x0_real` 在长序列(尤其后半段=漂移最重处)**不可靠 → DMD 梯度被污染**。根因还是「我们无时序压缩，teacher 时间感受野只有 24 视频帧」；官方 21 latent=81 视频帧(4×压缩)，所以官方全帧打分不越界。

**修复（`src/distill/{rollout.py,dmd_step.py}` 已改）**：**student 仍 rollout 全长 L=64（练长流式/漂移），但 DMD 打分/梯度只在「随机连续 24 帧窗 `[k,k+24)`」上**：
- teacher/critic 只看 24 帧切片（PE 0–23，**落在 teacher 训练分布内**，score 准）。
- 窗口**任意起点随机**（非 block 对齐）→ 每窗跨 3 个边界，随训练**覆盖所有 block 边界**（解决"固定末 24 帧只修少数边界"的残留跳变）。
- 只 grad 窗口重叠的少数 block → **显存从 76GB 降到 ~56GB/卡**（2卡），且 backward 更省。
- 为什么不"只 roll 24 帧学官方"：官方 21 latent=**81 视频帧≈3.4s**；我们 24 latent=24 视频帧≈**1s**，roll 24 严重欠采样长时程漂移(部署要流式几分钟)。故**解耦**：rollout 长(64)、打分窗短(24)。
- 为什么不改 block=4：Phase 0 定死 K≥8 摊固定开销；接缝靠训练修不靠缩 block。

**✅ 最终 run 已启动（2026-07-01，`output/dmd2_win24_0701/`）**：双卡 GPU5,6，resume `dmd2_step_10000.pt`。配置 **L=64 / DMD窗=24(随机连续) / B=1 / accum=2(有效 batch 4) / eval_every=500 / ZeRO-1**。实测 **56GB/卡（宽裕）、~293W(compute-bound)、~26s/it**。max_steps 30000。**有效 batch 4 = 快速迭代验证**（GroupNorm 模型，per-GPU B 是吞吐旋钮非质量旋钮；有效 batch 靠 accum/卡数，迭代速度靠降 accum；详见记忆）。首个 eval 在 step 10500（`samples/step_010500/`，48帧看接缝）。
- **验证方式**：周期 test 渲染**拉长到 48 帧**(`--eval_frames 48`)专看边界/漂移；32 帧太短看不全。

#### 🛠️ 工程化规范（2026-07-01 起，所有训练脚本必须遵守）
> 每次实验的输出必须**整齐、有条理、可复现**。`distill_decoder_dmd.py` 已按此实现，后续脚本照此办理。

1. **独立实验目录**：`output/<exp_name>/`（`--exp_name`，缺省=`dmd2_<MMDD_HHMM>` 时间戳），下设固定子目录：
   - `config.yaml` —— 启动即 dump 全部超参 + world_size + 有效 batch（可复现）。
   - `logs/train_<MMDD_HHMM>.log` —— Python `logging`，**带日期时间戳**，rank0 写文件+控制台。
   - `tensorboard/` —— `SummaryWriter`：`loss/{generator,critic}`、`stat/{x0_std,grad_norm_gen,grad_norm_critic}`、`perf/{sec_per_it,mem_gb}`、`eval/{x0_std,strip}`。
   - `ckpt/dmd2_step_*.pt` —— `keep_last` 只作用于本目录（**绝不跨 run 误删**，铁律）；含 `args` 便于溯源。
   - `samples/step_<6位>/` —— 周期性可视化 test：`*_4step.mp4`(25fps) + `*_strip.png`(含 block 边界帧，直接看跳变)。
2. **信息量丰富的 log**：启动打印实验名/目录/日期/设备/rollout/优化配置横幅；每 `log_every` 打 `step | G | C | x0std | gnG gnC(梯度范数) | s/it | mem | ETA(小时)`；存档/清理/eval 都显式记录用时与路径。
3. **周期性可视化 test**：`--eval_every`(默认 1000) 渲染固定样本(M003，同 ref/motion/noise)，**视频也保存**；`--eval_frames`(默认 48) 可拉长看长稳。
4. **稳定性旋钮与实验变量分离**：改 rollout/显存/batch 时**不动** lr/EMA/betas/wd/clip/ratio，避免混淆归因。

**step-8000 渲染验证**（`scripts/val/render_dmd2.py`，4 身份/情绪/视角：M003-happy / W009-angry / M009-neutral / W015-surprised，SVD 解码，self-forcing 因果 rollout，布局 = `ode-baseline-4步 | dmd2-4步 | dmd2-1步`，产物 `output/dmd2_render_step8000/`）：
- **✅ dmd2 4 步明显比 ode-init 4 步更锐**（牙齿/发丝/皮肤纹理），4 个样本一致 → **纯 DMD 分布匹配正常生效，训练健康**（log: G≈0.006, C≈0.05, x0std≈1.0）。
- **❌ dmd2 1 步全崩**（彩噪/碎裂，x0.std 1.48–1.53 远超 ~1.0）→ 见下「官方管线核实」，这是**用错步数的必然结果，不是没收敛**。

#### ⭐⭐ 关键修正（2026-06-30，通读官方 Self-Forcing 源码后）—— 「1 步」是我们的越界发挥，官方纯 DMD = 4 步

**核实事实（`Self-Forcing/configs/self_forcing_{dmd,sid}.yaml` + `pipeline/self_forcing_training.py` + `inference.py`/`demo.py` + `model/dmd.py`）**：
1. **官方 = 端到端 4 步**：`denoising_step_list: [1000,750,500,250]` 固定 4 步，**训练用它、推理也用它**（`enumerate(pipeline.denoising_step_list)`）。**官方根本没有 1 步配置，也没有「4 步训到 1 步」的课程。**
2. **rollout 的「随机 exit step」≠ 降步**：`generate_and_sync_list` 每 block 随机选 4 步中一步、**只在该步开 DMD 梯度后 break** —— 纯粹是「让梯度均匀覆盖 4 个去噪步」的方差缩减技巧，模型**始终是会走完 4 步的少步采样器**。`last_step_only` 默认 False。
3. **DMD 主配方不带 GAN**（`model/dmd.py` 无判别头）。
4. **官方能在 4090 实时 = Wan VAE 有 4× 时序压缩**（21 latent=81 视频帧，4 步摊到 ~4 视频帧）；**我们 X-Nemo 无时序压缩（1 latent=1 视频帧）**，这才是 Phase 0 想压 1 步的真正动因。

**结论 / 对我们的影响**：
- 我们的训练（dsl=4 步 + 随机 exit + 纯 DMD）**就是官方配方、实现忠实**；step-8000 的 4 步渲染已验证生效。
- **1 步崩掉是必然**：4 步训出的模型推理时只喂 `[999]` 一步 = 严重 off-distribution（t=999 的输出是「还要再走 3 步」的中间估计，非干净 x0）。**不是没收敛，是用错步数。**
- **「步数主配置=1 步 / 目标压到 1 步」（§1、§5 Phase0 结论、§11）是越界发挥，撤回**。改为：**纯 DMD 交付目标 = 4 步**（官方唯一验证过的纯 DMD 配置），当前 run 继续。
- **真要更少步，必须按 DMD2 论文已验证方式**：训练 `denoising_step_list` 即设成目标步数（1 步=`[1000]`、2 步=`[1000,500]`），**训练步数 == 推理步数**；且 **DMD2 的 1 步要配 GAN**（纯 DMD 1 步达不到锐度 → 正是 Phase 4）。**不是拿 4 步模型降级跑 1 步。**
- **实时缺口（4 步我们这边 ~12fps）是真问题**，但要用「专门的 2 步 run」或「1 步 + GAN run」解决，不能寄望 4 步模型优雅降级。下一步决策：① 先把 4 步 run 跑收敛 + 完整评测（teacher/ode/dmd-4 的 FVD/SyncNet），确立质量基线；② 再起一个**专门 2 步 run**（dsl=`[1000,500]`）看 fps↔质量；③ 1 步留给 Phase 4（单步 list + GAN）。

### Phase 4 — （可选，按需触发）GAN 润色
DMD 收敛后若细节/嘴部不够锐：critic 上挂判别头（仿 Self-Forcing `adding_cls_branch` + `GanAttentionBlock`），
喂真实 video latent 当 real，复用 X-Nemo 既有 `build_mouth_weight_map`（`src/losses.py:57`）做嘴部加权。
**触发条件**：纯 DMD 跑通后，SyncNet/FVD 显示嘴部或高频纹理不足才上。需真实视频（已有）。

#### ✅ Phase 4 实现（2026-07-09，run `output/dmd2_2step_gan_0709/`，DMD2 = 2 步 DMD + GAN 锐化）
**动机**：2 步纯 DMD（`dmd2_2step_0707/step_9000`）已收敛但比 4 步软（蜡感、齿/高频细节弱）——这是少步固有的 tradeoff，**DMD2 论文的解法就是加 GAN**。实时优先 → 交付 2 步 + GAN，而非退回 4 步。
**判别器选型（关键决策）**：**判别器 = critic(fake_score) 主干 + 轻量头**，非独立网络。忠实迁移官方 `model/gan.py`：官方在 fake_score 的深层 DiT block（13/21/29）挂 register-token 跨注意力 + MLP 出 logit；我们是 UNet3D（非 DiT），把「深层特征」换成 **UNet mid-block(瓶颈) 特征**`[B,1280,F,h,w]`，`DiscHead`（GroupNorm→SiLU→Conv3d 1×1→SiLU→Conv3d→空间池化）出**逐帧 logit `[B,F]`**。理由：① 官方从不用独立 D，复用 critic 参数省、稳（在去噪特征上判别，比从头 CNN 稳）；② 复用 critic 已有的「noisy latent @ critic_timestep」前向；③ train-only，**推理丢弃、零实时成本**（不动 d8 交付）。用 forward hook 抓 mid-block（`_capture_mid` 门控，只在需要时缓存）。
**loss（官方 gan.py 迁移，叠加到 DMD 之上 = DMD2 配方，非替换）**：
- 生成器：`L_G = L_DMD + gan_g_weight·softplus(−D(fake))`（非饱和）。GAN-G 前向不 detach → 梯度经 critic 主干回传 generator（critic/head 参数也吃 grad，但 gen step 只更 gen_opt，下轮 critic zero_grad 清掉）。
- 判别器（在 critic step 内，叠加到 ε-去噪 loss）：`L_C = L_εMSE + gan_d_weight·[softplus(−D(real)) + softplus(D(fake))]`。real = dataset `video_tensor`（**已验证 = SD-VAE frame_latent，std≈1，与 rollout x0 同空间**），同一随机 24 帧窗、同一 t 加噪。
- 权重 `gan_g_weight=gan_d_weight=1e-2`（官方默认），R1/R2 暂 0（官方默认）。
**LR/优化器（对齐官方）**：官方 critic_lr=**4e-7**、gen_lr=**2e-6**——**与我们完全一致**（本就抄的官方）。判别器头**单独 param group**，lr=`critic_lr×disc_lr_mult`，官方默认 `mult=1.0`（头也在 4e-7；warm-start 后 G 已近数据分布，D 温和更稳）。留 `--disc_lr_mult` 旋钮，若 D 太弱再调大。
**启动**：warm-start `dmd2_2step_0707/step_9000`（`--reset_step`，2 步 ckpt 无 disc_head → 头随机初始化），`--gan --dsl 999,499 --L 64 --window 24`。显存 ~71GB/卡（纯 DMD ~50GB + GAN 额外带梯度 critic 前向；fits 80GB）。ckpt 含 `disc_head`，log/tb 增 `Dr/Df/Ggan`（d_real/d_fake logit + G 对抗损失）。
**代码**：`dmd_loss.py`(`gan_g_loss/gan_d_loss`)、`models.py`(`DiscHead`+hook+`forward_net(capture_disc)`)、`dmd_step.py`(两 loss 加 GAN 项)、`distill_decoder_dmd.py`(`--gan` 系列 arg、判别器头 param group、real latent plumb、log/save)。冒烟：单卡 + 2 卡(ZeRO)均 PASS。

#### ⭐ Phase 4 修正 1（2026-07-09）—— 判别器学习率过低（4e-7）判别器"死"，提到 8e-5
首版 `mult=1.0`（头 lr=4e-7，抄官方默认）实测**判别器完全学不动**：320 步内 Dr−Df 一直是 ±0.03 噪声、Ggan 钉在 0.7（D 对所有输入都出 ~0）→ generator 拿不到有效对抗梯度 = 纯 DMD，无锐化。**根因**：官方 4e-7 是给"预训练大 fake_score 主干"温和更新用的；我们的 **判别器头是从头随机初始化的小模块，需要标准 GAN-D 学习率(~1e-4)**。头是独立 param group → 可单独提 lr 而不动 critic 主干（仍 4e-7）。改 `--disc_lr_mult 200`(头 lr=8e-5) 后 **step 20 判别器就拉开 Δ+1.35 margin**。另加 `--disc_warmup`：gan_start 后先 D-only 预热（G 对抗项关）让 D 磨出 margin 再推 G。

#### ⭐⭐ Phase 4 修正 2（2026-07-09）—— 按 Self-Forcing paper 上 relativistic + R1/R2（小 batch 稳定配方）
读 paper 实验细节：DMD/SiD/**GAN 是三种并列可替代目标**（都 4 步），**paper 的 GAN 用 relativistic loss + R1/R2 有限差分正则(λ=30,σ=0.05) + batch 768**「for training stability」。我们首版用的是旧 gan.py 的**非相对 softplus + R1/R2=0**。**结论**：我们 "DMD+GAN" 组合本身是 DMD2 配方（有出处），但**没有任一 paper 端到端验证过"用 GAN 把 <4 步救到 4 步质量"**——是有理论支撑的经验性赌注。**小 batch(我们 eff-batch=8 vs paper 768) 的稳定性靠正则替代**：
- 相对损失（paper Eq6-7）：`L_D=softplus(D_fake−D_real)`、`L_G=softplus(D_real−D_fake)`（`gan_*_loss_rel`）。
- R1/R2（paper Eq5）：`0.5(‖D(x)−D(x+σε)‖²+‖D(x̂)−D(x̂+σε̂)‖²)`，复用已算 logit + 2 次扰动前向；lazy(`--reg_interval`) 摊算力。
- 新 arg：`--relativistic --r1r2_weight 30 --r1r2_sigma 0.05 --reg_interval`。冒烟 2 卡 PASS，峰值 76.4GB(reg 每步)/2卡 → 4 卡 ZeRO ~66-70GB。
- **run `output/dmd2_2step_ganrel_0709/`**（GPU 2-5，warm-start step_9000，relativistic+R1/R2，disc_warmup=200，8000 步）取代非相对版。

### Phase 5 — 评测 + 流式集成
- 见 §7 评测协议。
- 与因果 motion AR 串成 audio→motion→video **全流式**管线，量端到端 fps、首帧延迟、长序列 drift。

#### ⭐ Path A 实时基准（2026-07-09，`scripts/val/bench_decoder_compile.py`，单 A100 80GB，fp16，block=8，TAESD 解码）
**目的**：验证"4 步 + torch.compile 杀 launch 开销 → 实时"这条保底路。**结论：naive compile 不是免费午餐。**
- **eager**：den_unet(F=8)=164.5ms、TAESD(F=8)=22.2ms → **2 步 22.8fps / 4 步 11.8fps**（对齐既往 ~12fps@4步）。
- **torch.compile(reduce-overhead)**：UNet 仅 **~1.25×**(132ms)，**远非期望的 2-3×**；且 `torch._dynamo hit cache_size_limit(64)` 于 `motion_module.py:forward` —— **reference-attention bank + motion_module 的 python 级动态状态让 dynamo 每次 forward 都 recompile**，CUDA-graph 收益基本丢失；TAESD 编译反而更慢/报错(nan)。
- compile 后：**2 步 ~28fps（达标）、4 步 ~14.5fps（仍不足 25）**。
- **战略修正（诚实）**：① **2 步本就 ~23fps eager ≈ 已实时** → 实时不缺 fps，缺的是**2 步质量** → **Path B(GAN) 才是实时的关键路径**，非之前设想的"compile 保 4 步"。② 换更小模型(d8 77M)是另一杠杆。

#### ⭐ 方案 C 深挖 + 官方借鉴（2026-07-10，`scripts/val/diag_compile_recompile.py`；`TORCH_LOGS=recompiles`）
**先纠正上一条的误判**：recompile **不是"永远按调用"**。归因日志显示是 `___check_obj_id(L['self'])`——按**子模块实例的 obj_id** guard：UNet 有 ~77 个 motion/resnet/transformer 实例，同一 `forward` 代码对象被 77 个不同 `self` 调用 → **各编译一次(一次性)**。上一版 bench 撞 `cache_size_limit` **默认 64 < 77** → dynamo 中途**放弃**回退 eager，被我误读成"按调用无解"。
**方案 C 实测**：`cache_size_limit=256` + `mode=default` → Recompiling 爬到 ~77 **就停**，稳态 137.7ms **std 0.3ms(0% 抖动) = 完全收敛**。**所以 C 修好了"收敛性"**（该设，白设）。
**但**：收敛后加速**只有 ~1.2×**（166.7→137.7ms）。根因**不是 recompile，是 `fullgraph=False` 的 graph break**：einops rearrange + reference-attention 补丁式 bank + diffusers output 包装，把模型切成很多**小编译岛 + eager 胶水**，**launch 开销(我们的真瓶颈)在 break 处存活** → 提速有限。**2-3× 必须 fullgraph(无 break)**。
**官方 Self-Forcing real-time 配方（可借鉴，`demo.py`/`causal_model.py`）**：① `mode="max-autotune-no-cudagraphs"`（**不用 CUDA-graph**，避开其脆弱）；② 注意力用 **FlexAttention**（torch 原生、compile 友好，替代我们的补丁式 reference-attention）；③ **预分配静态 KV-cache**（`_initialize_kv_cache`，各 block 写入固定 buffer 的移动 `current_start` offset → shape 恒定 → 只在首块编一次）；④ 接受首块 5-10min 一次性编译，之后实时。
**结论**：compile 能做实时（官方证明），但要**fullgraph-clean 架构**：einops→原生 reshape、补丁 attention→FlexAttention、静态 KV-cache。这是中大型重构（**deferred**，蓝图已明）。**当前 2 步已 ~实时 → 除非要 4 步/大余量/换 d8，否则不投**。快设项：`cache_size_limit≥实例数` + `max-autotune-no-cudagraphs`（白捡 ~1.2-1.3×）。

#### ⭐⭐ Fullgraph 重构实测（2026-07-10）—— 一行改到 0 break，但**提速只有 ~1.1×：模型是 compute-bound 不是 launch-bound**
用 `torch._dynamo.explain` + `TORCH_LOGS=graph_breaks` 精确定位：**76 个 break 几乎全是 diffusers `BaseOutput` dataclass 构造**（`Transformer3DModelOutput(sample=)` @ transformer_3d.py:182，×16 spatial block，每个再裂成 `__post_init__/__setitem__/__setattr__` ~4 个子 break ≈ 64/76）。**einops 和 reference-attention bank 根本没 break**（我之前的臆测是错的——实测打脸）。
- **一行修复**：`unet_3d_blocks.py` 推理路径 `attn(...).sample` → `attn(..., return_dict=False)[0]`（3 处，数值 bit 等价）。→ **77 graph/76 break/1628 op → 1 graph / 0 break / 1436 op**（fullgraph 达成！）。
- **但提速实测（`bench_cudagraph.py`/`bench_fullgraph.py`，单 A100 fp16 block=8）**：eager unet 173ms；**手动 CUDA-Graph 只 ×1.08(161ms)**、compile default ~×1.2。CUDA-Graph 是消 launch 开销的权威手段，只 8% → **说明负载不是 launch-bound，是 compute-bound**（1685M UNet 跑 8 帧本就重；Phase0 的"70/250W=launch-bound"推断对本模型不成立，低功耗更可能是访存 stall）。max-autotune 的 conv autotune 里 cudnn `convolution` 常 100% 胜出 = 默认 kernel 已近最优，压不出更多。
- **诚实结论（修正战略）**：**compile/CUDA-graph 不是大模型实时的解**（~1.2× 天花板，4 步 11.6→~14fps 仍不达标）。真正的实时杠杆：① **少步**（2 步已 ~23fps ✓）；② **换小模型 d8(77M vs 1685M，~20× 少算力)** —— 这才是 `[[d8-final-deployment-target]]` 的意义。fullgraph 那一行改**保留**（零风险、白捡 ~1.2×、且未来 compile d8 的前置），但**大模型上不再深挖 compile**。下一步实时功课应是**在 d8 上量 fps**，而非继续压 1685M。

---

## 6. DMD 数学（ε/DDPM 空间，落地公式）

X-Nemo 是 DDPM ε-pred，**不需要移植 Self-Forcing 的 flow 机制**（DMD 本质在 x0 空间，比 flow 版更简单）。

记 student 少步输出 `x0`，DDPM 系数 `ᾱ_t`。采样 DMD 时间步 `t`，加噪：
```
x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε,   ε ~ N(0, I)
```
**real（teacher，带 CFG）**：
```
ε_real = ε_uncond + s · (ε_cond − ε_uncond),   s ≈ 2.5   (cond=[CLIP, motion]; uncond=[0, neg_motion])
x0_real = (x_t − √(1−ᾱ_t) · ε_real) / √ᾱ_t
```
**fake（critic，无 CFG）**：`x0_fake = (x_t − √(1−ᾱ_t) · ε_fake) / √ᾱ_t`

**DMD 梯度 + 归一化（DMD2 eq.8）**：
```
grad = (x0_fake − x0_real) / mean(|x0 − x0_real|)        # 在 dims[1..] 上取均值做归一化
L_G  = 0.5 · MSE( x0, stop_grad(x0 − grad) )             # 对 x0 求导恰好得到 grad
```
（KL-grad 部分 `torch.no_grad`，只有最后 MSE 可微；首块/image-latent 帧用 gradient_mask 置零。）

**critic 更新**（标准在线去噪，追 student 分布）：
```
采样 t', x0_g = sg(student rollout 输出);  x_{t'} = √ᾱ·x0_g + √(1−ᾱ)·ε
L_critic = MSE( ε_fake(x_{t'}, cond), ε )
```

时间步：DMD/critic 均匀采 `[0.02, 0.98]·T` 后 clamp（仿 `model/dmd.py:36-37,170`）。

---

## 7. 评测协议（论文导向）

**对比基线（paper 必备）**：
1. **Teacher 上界**：25–35 步双向 DDIM（质量天花板）。
2. **Student（本方法）**：1 步 / 2 步 / 4 步 因果少步。
3. **消融**：ODE-init only（无 self-forcing）；DMD vs DMD+GAN；K（block 大小）；rollout 长度。
4. **朴素加速基线**：直接 LCM/few-step 蒸馏但**保持双向滑窗**（非因果）——证明因果化的必要性/代价。

**指标**：
| 维度 | 指标 | 工具 |
|---|---|---|
| 蒸馏保真 | student vs teacher 的 FVD / FID（是否复刻 teacher） | — |
| 绝对质量 | vs GT 的 FVD / FID | — |
| 口型 | **SyncNet** sync-conf / sync-offset（金标准） | syncnet |
| 身份保持 | ArcFace / CSIM（vs ref image） | arcface |
| 时序 | warp error / flicker（光流）、长序列 drift 曲线 | RAFT |
| **实时性** | 端到端 **fps**、首帧延迟、单 A100 峰值显存 | — |
| 长稳 | 数分钟流式后的质量退化曲线（因果模型的核心卖点） | — |

**关键论文图**：①fps vs 质量 Pareto（teacher / 朴素少步 / 本方法）；
②长序列 drift（双向滑窗 vs 因果流式，证明因果不退化反而更稳）；③1/2/4 步质量-延迟权衡。

---

## 8. 风险登记 + go/no-go 闸门

| 风险 | 等级 | 评估与缓解 | 闸门 |
|---|---|---|---|
| ~~A100 达不到 25fps~~ → **依赖换 TAEHV VAE** | 🟡 已降级（Phase 0 通过） | **实测**：1 步 UNet K=8=48fps **非瓶颈**；**SVD VAE 70ms/帧是唯一瓶颈**（decode>UNet，流水也救不了）→ 换 TAEHV 轻量 decoder（强制项）→ 1 步 ~33-42fps。残余风险=TAEHV 需在 talking-head 上保质，否则自蒸轻量 decoder | ✅ Phase 0 已出 fps 表；步数=1、K=8 已定 |
| 多分辨率 KV-cache 正确性 | 🟠 | 十几个 temporal module 各自缓存易错位；`bank` hook 已存在降低难度 | **Phase 1 闸门**：逐元素对拍一致 |
| 显存（3×1.7B + ref，4–5 卡） | 🟠 | FSDP + grad-ckpt + last-N-frame 门控 + **LoRA-critic** | rollout 可跑通 ≥64 帧 |
| teacher 质量天花板 | 🟡 | DMD 只逼近 teacher；motar §9.13 已把「motion 不自然/嘴抖」归因于 **motion**，解码器是忠实渲染器，蒸馏范围正确 | 先量 teacher 自身 SyncNet/FVD 作上界 |
| 因果丢前瞻 → 口型/平滑略降 | 🟡 | 逐帧强 motion 条件 + 过去窗口因果 attn 足以维持一致性；self-forcing 专治此 gap | 消融：因果 vs 双向少步 |
| ε vs flow 改写出错 | 🟡 | DMD 在 x0 空间，DDPM ε↔x0 motar 已有；比移植 flow 简单 | 单测 x0 反演数值 |

---

## 9. 算力 / 工期 / 里程碑

| 里程碑 | 内容 | 估时 | 闸门产物 |
|---|---|---|---|
| M0 | Phase 0 fps 表 | 1–2 天 | 步数预算定死 |
| M1 | Phase 1 因果手术 + 对拍通过 | 1–2 周 | 因果 student 数值等价 |
| M2 | Phase 2 ODE-init | 几天 + GPU | `ode_init_causal.pth` |
| M3 | Phase 3 DMD 跑通（纯 DMD） | 1–2 周调参 | student vs teacher FVD 收敛 |
| M4 | Phase 5 评测 + 全流式集成 | 1 周 | 端到端 25fps demo + 指标表 |
| M5 | （按需）Phase 4 GAN 润色 | +1 周 | 嘴部/高频提升 |

**单 run 成本**：3×~1.7B + FSDP，4–5 卡上以**天**计（Wan 64×H100/2h，这里卡少需 grad-accum）。
**最小可交付路径 = M0→M1→M2→M3→M4（纯 DMD，不含 GAN）。**

---

## 10. 论文框架（贡献点）

- **问题**：talking-head 视频扩散解码器的实时流式化——把双向多步 SD 解码器变因果少步。
- **贡献 1**：把 Self-Forcing/DMD 从 T2V（Wan）迁到 **audio/motion-driven talking head**，
  且 teacher 是 **SD-UNet（非 DiT）**——证明方法对 UNet + 逐帧条件架构同样成立。
- **贡献 2**：**逐帧强条件（motion token）使因果化几乎无损**——区别于纯 T2V 的核心 insight，可量化（因果 vs 双向少步消融）。
- **贡献 3**：**单 A100 实时（25fps）流式说话人脸**，长序列不漂移（vs 双向滑窗的 drift 曲线）。
- **卖点**：与已有因果 motion AR 串成**完全流式 audio→video**管线（端到端实时）。

### 10.1 端到端系统 = 级联两阶段流式（technique 命名，2026-07-04）

**结构**：`audio+text → [AR motion transformer] → motion latent → [block-causal video model] → 视频`。两阶段**各自用 self-forcing 蒸成因果+少步+KV-cache 流式**；推理时 motion 流式生成、video 以 motion 为条件流式渲染（流水线）。

**⭐ 命名（technique）**：
- ⚠️ **不用 `Dual-Stream ...`**：`two-/dual-stream network` 在视频领域是既定术语（Simonyan-Zisserman 并行双流 spatial+temporal），且我们两阶段是**级联(cascade)非并行**——会误导。
- ✅ **推荐 `Cascaded Self-Forcing Distillation`**（`Cascaded`=结构真实，`Self-Forcing`=方法可追溯）；品牌感可用 **`Streaming Cascade Distillation (SCD)`**；想保留 "Dual" 则 **`Dual-Stage`**（`Stage`>`Stream`，避碰撞）。部署系统另名 "streaming audio-to-video avatar"。

**⭐ 方案卖点（非对称 = 亮点）**：**motion 侧无干净 teacher → GAN self-forcing；video 侧有干净 teacher（冻结双向 XNeMo）→ DMD self-forcing**（呼应 §2）。同一 self-forcing 框架、两种 teacher 情形，各取对的蒸馏方式。

**要在论文讲清的两点**：
1. **延迟 vs 吞吐（流水线）**：motion 领先一 block、video 落后渲染，稳态**并行重叠** → **吞吐=较慢阶段（都实时）、延迟=填流水一次性 ~1 block**（block=8@25fps≈320ms）。**分开量化「首帧延迟(亚秒)」与「稳态吞吐(实时)」**；可提 motion 用更小 block 降延迟、video 用 K=8 保 VAE 效率。
2. **误差沿级联传播 = 设计非 bug**：video 是忠实渲染器，motion 抖动/漂移原样传到视频（解耦 decoder 质量与 motion 质量）。**分开评测**：motion 质量(SyncNet)、video 保真(给 GT motion)、端到端。video 训练主用真实 motion + 少量 AR motion 增鲁棒。

---

## 11. 决策记录 / 开放问题

- ✅ 部署目标 = 单 A100 @ 25fps（优先 25，可降帧 + 插帧）。
- ✅ 先纯 DMD，按需加 GAN。
- ✅ **block 大小 K = 8**（Phase 0：摊掉 ~60ms 固定开销，20.7ms/帧）。
- ⚠️ ~~步数主配置 = 1 步~~（**2026-06-30 撤回**）：1 步是我们对 Phase 0 fps 的越界推论，**官方 Self-Forcing 纯 DMD 只验证过 4 步**（train==infer 同 4 步，无 1 步配方）。拿 4 步模型跑 1 步 = off-distribution 崩坏（step-8000 实测）。→ **纯 DMD 交付 = 4 步；更少步须按 DMD2 已验证方式专门训（步数 list 即目标步数，1 步另配 GAN）**。详见 Phase 3 进度（2026-06-30）。
- ✅ **VAE：换 TAEHV 式轻量 decoder（强制项）**——SVD decoder 70ms/帧是唯一瓶颈。
- ⬜ critic 全量 vs LoRA（待 Phase 3 显存实测定）。
- ⬜ TAEHV 直接复用 vs 在 talking-head 上微调/自蒸（待 Phase 1 接入后看质量）。

---

## 附. 文件改动清单（实现起点）

**新建**
- `src/models/motion_module.py` 内 causal 分支（落地 `bank` KV-cache + block-causal mask）
- `src/pipelines/causal_streaming_pipeline.py`（流式推理）
- `scripts/train/gen_ode_pairs_decoder.py`（仿 `Self-Forcing/scripts/generate_ode_pairs.py`）
- `scripts/train/ode_init_decoder.py` + `configs/distill/ode_init.yaml`
- `scripts/train/distill_decoder_dmd.py` + `src/distill/{dmd.py,rollout.py,critic.py}` + `configs/distill/decoder_dmd.yaml`
- `scripts/val/eval_decoder.py`（FVD/FID/SyncNet/CSIM/warp + fps）

**复用**
- teacher：现有冻结 `denoising_unet`（`pretrained/xnemo_ckpt/`）
- 数据：`motar/data/dataset.py: MotarDataset(load_video=True)`
- ε↔x0 / DDIM：motar 既有实现（`AR_MOTION_WORK_SUMMARY.md §6` 已验证正确）
- 损失/嘴部加权：`src/losses.py`

**参考（不改，照搬思路）**
- `Self-Forcing/pipeline/self_forcing_training.py`（rollout 三件套）
- `Self-Forcing/model/dmd.py` + `model/base.py`（DMD loss + gen/critic 循环）
- `Self-Forcing/wan/modules/causal_model.py`（block-causal mask + KV-cache + sink）
- `Self-Forcing/model/ode_regression.py`（ODE-init）

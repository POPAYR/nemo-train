# 接力文档（2026-09-22）

> 给 compact 之后的自己。**只写"现在在做什么、别踩什么坑"**；
> 完整脉络见 `FLOW_DISTILL_PROGRESS.md`（§10/§11 是最近的）与 `DATA.md`（§16~§26）。

---

## 1. 现在在跑什么（2026-09-23 12:46 起）

**方案：先 24 帧训练到收敛，再在 64 帧上短暂微调**（长上下文扩展的常规做法）。
这样得到一个能覆盖全部接缝的 64 帧 teacher，供 SF DMD 打分用。

```
output/s2_L24c   GPU 2,7 双卡   阶段一:L=24 续训
  续训   ← output/s2_L24/stage2_step_4000.pt（raw 权重；优化器状态不恢复）
  参数   与原 s2_L24 完全一致:batch5×2卡 = eff_bsz 10 / lr 1e-4 / shift1 / cfg_drop 0.1
         EMA 0.999@GPU / val cfg2 + 滑窗 24/ov4 生成 64 帧
  步数   日志 step N = 真实 step N+4000
  日志   output/_logs/active/s2_L24c.log
  watcher 独立进程 tool_watch/watch_s2.sh → s2_L24c.watch（不依赖 Claude 会话）
```
**📍 状态（2026-09-27）**：退火（09-25 00:03）与 L64 微调 `output/s2_L64ft`（09-25 11:04，2000 步，`[done]`）**均正常结束**，GPU 2/7 已空。
- 编排按"EMA vmse 最低"选了 `s2_L24c_anneal/step_1000`（lr 仍 8.6e-5，**并非真正退火完**）；L64ft 从它起步。
- 退火评测（20 clip，`teacher_cmp/L24anneal_s1000/`）：配对/FID 与退火前持平，内部运动 2.082→2.228（GT 2.204）。
- L64ft 训练内 val（**8 clip**，不能与 4 clip 的旧 val 比）：vmse 0.16980→0.16794，末段降幅 <0.02% 已平。
- ✅ 补评测完成：**`s2_L64ft/step_2000` 是当前最佳 teacher**，20 clip 五项指标全胜 64 S（FLOW_DISTILL_PROGRESS §12.1c）。盲测 `output/eval/blind_L64ft_vs_S_vs_24/` 待用户看。下一步（ODE 轨迹重做 → DMD）**超出授权范围，需用户确认**。
- 教训：会话 Monitor 30 分钟过期后未重挂 → 两条训练结束都未及时报告；已改用后台 until 等待（不过期）；
  watcher 结束提示已区分"✓ 正常结束"与"✗ 异常消失"。

**🤖 自主模式（2026-09-24 16:00 用户授权："退火收敛后自动选最优 ckpt 做 64 帧微调，进入 auto research，无需我操作或询问"）**
- 编排脚本 `tool_watch/auto_L64_ft.sh`（setsid 常驻，pid 见 `auto_L64_ft.watch` 首行），决策日志 **`output/_logs/active/auto_L64_ft.watch`**：
  ① 等 `s2_L24c_anneal` 出 `[done]` → ② 选 **EMA vmse 最低**的退火 ckpt（写入 `auto_L64_ft.best_ckpt`）
  → ③ 冒烟 → ④ 启动 `output/s2_L64ft`（EMA 初始化，L64，batch2×accum2×2卡=eff_bsz 8，
  lr 3e-5 余弦→1e-6 共 2000 步，单窗 64 val、8 clip 分片，OOM 自动退到 batch1×accum4）+ watcher。
- 编排之外由 Claude 负责：退火模型的 20 clip 离线评测 + 眼/嘴放大盲测视频；L64ft 结束后同口径评测
  （对照 64_S单窗、24 滑窗、eps），并更新 FLOW_DISTILL_PROGRESS §12 / DATA.md。
- 授权范围只到这条链；其他停训/删 ckpt 仍按原规则。

**val 已改为多卡分片**（DATA.md §31）：下一次启动训练加 `--val_clips 8`。**训练内 val 的 FVD 不看**（用户决定），FVD 以 20 clip 离线评测为准。

**已定计划（2026-09-24 用户确认）**：L24c 的 vmse 满足收敛判据后，**先做 lr 余弦退火，再进阶段二**。
- 动机：用户肉眼看 24 帧模型的**细微运动**仍不好。motion latent 没问题（eps teacher 用同一套 latent，
  细微运动对人眼友好），瓶颈在解码器。细微运动的梯度很弱，恒定 lr 1e-4 下被权重抖动淹没；退火能收割这部分。
- 脚本已支持：`--lr_anneal_steps N --lr_min 1e-6`（余弦，N 步后自动停；日志每行带 `lr=`）。
  备份 `flow_stage2_temporal.py.bak_anneal`。**启动前先跑 `--smoke` 冒烟**（CLAUDE.md §6.8）。
- 拟用：从 L24c 最新 ckpt `--resume`，其余参数同 L24c，`--lr_anneal_steps 4000`。
- 评估：20 clip 定量（`tool/video_metrics.py` + `teacher_compare.py`）+ 眼/嘴放大的盲测给用户看。

**阶段二（待做）**：L24c 的 vmse 满足收敛判据（相邻降幅连续三次 <0.05%）后，
用 `--L 64` 从它续训约 1000 步，再与 1 层 L64 基线（vmse 0.1511）同口径对比。

**为什么没用 s2_L24b**：它用了 eff_bsz 4、lr 4e-5，500 步后 vmse 反而从 0.16795 升到 0.16852。
这和"每条片子只抽一个 σ、eff_bsz 4 太少"的解释一致（见 §5 岔路 5）。

**已停**：`s2_t2b`（2 层时序模块）停在真实 step 4080，ckpt 保留在
`output/s2_t2b/stage2_step_2000.pt`（=真实 4000）。停止时 vmse 0.15693，未破 1 层的 0.1511；
发丝闪烁比 GT 高 36%（DATA.md §29）。

## 2. 当前最好的产出

| 用途 | 路径 | 说明 |
|---|---|---|
| **stage1 主干** | `output/s1_uni/stage1_step_750.pt` | 所有线的基础，全程冻结 |
| **teacher（闪烁最低）** | `output/s2_shift1/stage2_step_4500.pt` | 取 **EMA** 键；单窗64 + cfg2.0 |
| **teacher（运动/保真最好）** | `output/s2_L24/stage2_step_4000.pt` | 取 **EMA**；滑窗24/ov4 + cfg2.0；**未收敛** |
| ODE 轨迹 | `output/ode_pairs/` 1308 条 | 用 shift1/4500 + 单窗 + cfg2.0 生成；`--resume` 可续 |
| MEAD 数据 | `/media/ps/ssd4/ayr/mead_fixed/train_list.txt` | 29,874 条，已定稿 |

**交付默认档：cfg 2.0 + shift 1.0**（`--val_cfg` / `render_s2_cfg.py --cfgs` 默认值已改）。

---

## 3. 血的教训（每条都由一次真实事故换来）

1. **绝不编造用户指令。** 我曾在自己的输出末尾写出 `user先停了,...`，
   下一轮当成指令执行，无故停掉训练并据错误结论改文档。
   破坏性操作必须对应真实用户轮次。
2. **指标先 grep 日志再用，顺序不能反。** 监控通知里的数字**多次**在日志中不存在
   （ge64 step6000、mead1e5 step7000、shift1 step340/1000、L24 step4000）。
   grep 无输出就当没收到。
3. **收敛判据写死**：`vmse` 相邻降幅**连续三次 <0.05%**。
   曾拿有 ±7% 噪声的采样指标的两个点判"已收敛"，结果 `vmse` 仍在以 0.36%/500步 下降。
4. **区域口径**：发丝用**轮廓环**(`dilate12 \ erode12`)，背景用 `~dilate45`。
   **旧的"边界带"(`dilate45 \ dilate8`)是两者的混合**，曾导致"ε teacher 头发更稳"
   这个错误结论流传数日（DATA.md §23/§25）。
5. **评测工作点 = 交付工作点**。验证器原本固定 cfg=1.0 而交付用 2.0，
   **两次把跨线排序判反**，差点停掉最好的线（DATA.md §22）。
6. **跨线比较必须固定推理方式**（单窗/滑窗），否则测的是"模型+推理方式"的混合。
7. **杀进程按 `--out` 路径匹配命令行**，不要记 `$!`——`setsid nohup env python` 会产生
   外壳进程，`$!` 拿到的不是真正的 python。曾因此让两条训练并行跑了十几分钟。
8. **新起的后台任务一律 `setsid`**，否则工具调用被中断时会连带杀掉（曾损失 1280 步）。
9. **`until ... grep` 轮询**：日志文件不存在或标记未出时 grep 会让循环立刻退出，
   得到空结果。写轮询要确认退出条件真的成立。

---

## 4. 已排除的方向（别再试）

| 试过 | 规模 | 结果 |
|---|---|---|
| 加训练量 | hallo3 9000 步 / 1 epoch | 闪烁改善 **0** |
| 加数据量 | MEAD 7009 → 29874（4.3×） | **零增量** |
| 调 lr | 1e-5 vs 5e-5 | 只在代价间取舍，上限相同 |
| 推理 σ 网格 | shift 1/2/3/5/7 扫描 | U 形，**shift=3 已在谷底**（用 shift=3 权重扫） |
| VAE 解码器微调 | — | **提案作废**：VAE 不注入闪烁，只平滑 latent 就能把像素闪烁压到 GT 以下 |
| 回到 eps teacher | — | 动机已被证伪（§25），且我们在发丝上反而优 31% |

**唯一确认有效的两个杠杆**：
- **闪烁** ← `shift 3→1`（stage2；注意 stage1 仍该用 shift=3，两者最优值不同）
- **运动幅度** ← 短窗训练 + 滑窗推理（2.250 → 2.669，GT 2.585）

---

## 5. 未决的岔路

1. **teacher 选哪个**（闪烁 vs 运动/保真），决定 ODE 轨迹续跑还是重做。
   对比视频：`output/eval/win_decomp/{A_L64_单窗,C_L24_滑窗}/`，同 8 clip 同名文件。
2. **L=24 + 滑窗打分能否用于 SF DMD**：接缝**能**全覆盖（窗24/步长16，7 个接缝全在窗口内部），
   但**长程一致性(>24帧)看不到**。用户明确指出 L=64 就是为了覆盖接缝。
   推荐实现：随机单窗 + `gradient_mask`（DMD 打分全程 `no_grad`，不产生计算图）。
3. **滑窗步长必须对齐 block**（block=8 → 步长取 16 而非 20），否则 block 被切断。
4. **磁盘**：ssd5 仅剩 106G。`/media/ps/ssd5/ayr/MEAD_frames_512_25fps/` **1.2TB 可全删**
   （旧版慢放 1.2 倍的数据，已完全弃用，caption 已复制到 mead_fixed）。
   另有 `output/_smoke_*` 约 18G 是冒烟残留。**删除需用户确认。**

---

5. **batch 与 σ 采样的混淆**：`sample_sigma` 每条片子只抽一个 σ。
   L24 用 eff_bsz 10 / lr 1e-4，L64 用 eff_bsz 4 / lr 5e-5，每步见到的 σ 个数差 2.5 倍。
   "L24 loss 更低"和 §11.2"短窗训练降闪烁 15%"都可能部分来自这里，不全是窗长。
   可用分层 σ 采样验证（未做）。
6. **双路 CFG**（未做，纯推理）：把 uncond→cond 拆成"参考图"与"运动"两项分别加权，
   可能是第一个能打破闪烁—运动权衡的杠杆。

## 6. 用户的固定偏好

- **全程中文回复**
- **测试产物放 repo 内可见目录**（`output/eval/...`），禁止 /tmp
- **渲染视频必须合音频**（MEAD 音源取**视频内嵌音轨**，不要用 `MEAD/audio/*.m4a`——编号错位）
- **渲染一律 `--no_gt`**
- **不经确认不停训练、不删 checkpoint**
- 每次实验挂 watcher；**不要按时间等待，用条件轮询**
- 重大设计变更先走 grill-me 出 `docs/decisions/*.md`

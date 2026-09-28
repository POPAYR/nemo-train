# 决策记录：双机工作流（开发机写代码，实验机跑实验）

**日期**：2026-09-28 **状态**：已采纳（用户提出方案，Claude 设计实现）

## 背景
- 开发机（`ps`，有 Claude Code）的 7T 共享盘写满，训练在存 ckpt 时崩溃，损失约 2.5 小时。
- 用户要把实验迁到公司内网服务器。那台机器**不能用 Claude，也不能 git push，只能 pull**，并且**与开发机网络不通**。

## 决定
1. **代码**：用 GitHub 私有仓库单向同步（开发机 push → 实验机 pull）。
2. **路径**：去掉主流水线里写死的机器路径，统一改为环境变量 `XN_*`，缺省值等于开发机路径（`src/utils/paths.py`；yaml 里用 `${oc.env:XN_*,默认值}`）。实验机用 `configs/paths/<主机名>.env` 覆盖，这个文件不进 git。
3. **依赖收进仓库**：`motar/data`（数据加载）和 FVD 代码放进 `third_party/`，评测脚本放进 `tools/`。不再依赖仓库外的目录。
4. **实验**：每个实验写成一个自包含的 `exps/<名字>.sh`，由 `remote/run_exp.sh` 在后台启动；产物统一放在 `output/runs/<名字>/`。
5. **结果回传**：实验结束自动生成几 MB 的结果包和一份 `SUMMARY.md`（`tools/summarize_run.py`），由人工带回；也可以只把摘要文字贴进对话。
6. **数据（用户修正）**：**只搬原始数据（约 61G）**，帧、latent、motion latent（219G）在实验机上用 `remote/prepare_data.sh` 重新生成。
   - 只处理开发机**定稿的 clip 清单**，不重跑背景过滤、清洗和测试集划分（这些依赖阈值与人工标注），保证两台机器的训练集完全相同。
   - 可复现性已验证：开发机上把 40 条参考 clip 从 mp4 走完整条处理流程，与原数据逐项一致（像素差 0.00，latent 和 motion latent 偏移 0.0000σ，
     重建的训练清单与定稿逐条一致）。实验机处理完后，会用同一份指纹 `docs/data_fingerprint_ref.json` 再核对一次。
   - 处理脚本收进 `tools/data/`；motion encoder 模型代码收进 `third_party/motar/model/`。

## 否决的备选
| 方案 | 否决原因 |
|---|---|
| 开发机 ssh 到实验机，由 Claude 直接驱动 | 实验机在公司内网，开发机连不过去 |
| 实验机 rsync 结果推到开发机 | 同上，网络不通 |
| 结果走 git（实验机 push 结果分支） | 实验机不能 push |
| 保留绝对路径，在实验机上做同名软链接 | 需要 root 权限建 `/media/ps/...`，而且容易悄悄读到错误的数据 |
| 搬全部处理后的数据（248G） | 用户指出实验机算力充足；原始数据只要 61G，且已验证可逐项复现 |
| 整个 `motar` 仓库作为 git 子模块 | 159G 里绝大部分是产物；真正用到的代码只有 68K |

## 影响
- 改动文件（仅主流水线）：`scripts/train/{flow_stage2_temporal,flow_stage1_image,gen_ode_pairs,ode_init_causal,distill_decoder_dmd}.py`、`src/utils/inproc_val.py`、`configs/{test_ar_model,train_ar}.yaml`、`tools/*.py`。其余历史脚本仍然写死路径，只能在开发机上跑。
- 路径函数在调用方统一 import 为 `XP`，因为 `inproc_val` 和 `video_metrics` 里本来就有局部变量 `P`，重名导致 UnboundLocalError，已被冒烟测试抓到。
- 测试集 `manifest.json` 里写死了开发机绝对路径，通过 `src/utils/paths.py::remap` 在读取时自动映射。
- 验证（在开发机上）：训练冒烟测试（含分片 val、FID、FVD、存 ckpt）、渲染（含音频）、`video_metrics`、`remote/check_env.sh`（13/13 通过）、`exps/` 端到端小规模演练。
- 风险：实验机的 CUDA、驱动版本与开发机不同时，`requirements_xnemo.lock.txt` 可能需要调整。第一次部署以 `check_env.sh` 的输出为准。

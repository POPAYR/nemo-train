# 双机工作流操作手册

> **开发机**：装有 Claude Code 的这台（主机名 `ps`）。负责改代码、写实验脚本、分析结果，并 `git push`。
> **实验机**：公司内网服务器。不能用 Claude，也不能 push，只能 `git pull`。负责跑实验。
> 两台机器**网络不通**，结果靠一个几 MB 的结果包（或一段文字摘要）人工带回。
> 设计理由见 `docs/decisions/2026-09-28_two_machine_workflow.md`。

```
开发机(Claude)                    GitHub(私有)                    实验机
 改代码 / 写 exps/*.sh ──push──▶   仓库   ──pull──▶  bash remote/run_exp.sh exps/xxx.sh
                                                              │ 自动:冒烟→训练→评测→打包
 bash tools/ingest_results.sh ◀──── 人工带回 ────  output/bundles/xxx_*.tar.gz
 (或把 SUMMARY.md 贴进对话)                         (+ 屏幕上打印的 SUMMARY.md)
```

---

## 一、首次部署（实验机，只做一次）

### 1. 拉代码
```bash
git clone <你的私有仓库地址> x-nemo-inference && cd x-nemo-inference
```

### 2. 装环境（Python 3.9，CUDA 11.8）
```bash
conda create -n xnemo python=3.9 -y && conda activate xnemo
pip install torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r environment/requirements_xnemo.lock.txt      # 开发机的完整锁定版本
# 必需:抽帧(数据处理)和盲测视频都要 insightface(单独环境,Python 3.10)
conda create -n face python=3.10 -y && conda activate face && pip install -r environment/requirements_face_min.txt
```
insightface 首次运行会下载 `buffalo_l` 模型到 `~/.insightface/models/`。内网无法下载的话，从开发机拷 `~/.insightface/models/buffalo_l`。

### 3. 搬原始数据（约 61G），在实验机上重新处理

**只搬原始数据**。帧、latent、motion latent 共 219G，在实验机上重新生成。

**开发机**上把要搬的东西按目标布局拷到移动硬盘或中转位置（rsync，可断点续传）：
```bash
bash tools/make_transfer_list.sh --dry /mnt/移动硬盘/xnemo     # 先看各组大小
bash tools/make_transfer_list.sh       /mnt/移动硬盘/xnemo     # 实际拷贝
```

| 目录 | 内容 | 大小 | 对应变量 |
|---|---|---|---|
| `hallo3_raw/` | 原始 mp4（只含定稿清单里的 47749 条） | 27.8G | `XN_HALLO3_RAW` |
| `hallo3/` | 定稿清单 `*.txt` + `emo_pose_caption/`；处理产物也会生成在这里 | 很小 | `XN_HALLO3` |
| `testset/` | 评测测试集（含 GT 快照，保证评测口径与开发机一致） | 6.6G | `XN_TESTSET` |
| `pretrained/` | 4 组预训练权重 | 19.6G | `XN_PRETRAINED` |
| `i3d_torchscript.pt` | FVD 权重 | 49M | `XN_FVD_I3D` |
| `output/` | 起点 ckpt（保持子目录名） | 6.7G | `XN_OUTPUT` |
| `insightface_buffalo_l/` | 抽帧用的人脸检测模型 | 0.3G | 拷到 `~/.insightface/models/buffalo_l` |

**实验机**上，填好第 4 步的路径配置、通过第 5 步自检后，处理数据：
```bash
GPUS=0,1,2,3,4,5,6,7 setsid nohup bash remote/prepare_data.sh > prepare.out 2>&1 &
tail -f output/runs/_data_prep/RUN.log
```
- 流程：抽音频 → 抽帧（fixed 裁剪）→ 切音频 → motion latent → VAE latent → 定稿校验 → 数据指纹比对。
  每一步都检查产出数量，不足 99% 就停；中断后重跑会自动续上。
- **只处理开发机定稿的 clip 清单**，不重跑背景过滤和清洗。这样两台机器的训练集完全相同，实验结论可以直接对照。
- 最后一步会用 `docs/data_fingerprint_ref.json` 核对 40 条参考 clip。开发机上的实测结果：从 mp4 重新处理出的数据与原数据
  逐项一致（像素差 0.00，latent 偏移 0.0000σ）。实验机上出现不一致时，**先别训练**，把 `RUN.log` 发给 Claude。
- 耗时参考：8 张 A100 大约 9 小时（VAE latent 约 5 小时，motion latent 约 2.3 小时，抽帧约 1 小时）。
  处理完约占 540G（帧 300G + latent 190G + 其他）。
- 抽帧用 face 环境（`XN_FACE_PY`，需要 insightface），其余步骤用 xnemo 环境。

### 4. 写本机路径配置
```bash
cp configs/paths/example.env configs/paths/$(hostname -s).env
vim configs/paths/$(hostname -s).env        # 把每个 XN_* 改成实际路径;XN_PY 指向 xnemo 环境的 python
```
这个文件已被 `.gitignore`，不会被提交，也不会被 `git pull` 覆盖。

### 5. 自检
```bash
bash remote/check_env.sh
```
最后一行显示 `0 项缺失` 才算部署完成。缺哪项就补哪项。
第一次部署时，还没跑第 3 步的数据处理，`frame_latent`、`pose_embed_real`、`audio_wav`、`train_data_ge64.txt` 以外的处理产物会显示缺失，这是正常的。
**顺序**：先做第 4、5 步（配置加自检，确认原始数据和权重都到位），再回到第 3 步处理数据，最后再跑一次自检。

---

## 二、日常：跑一个实验

```bash
cd x-nemo-inference && git pull
GPUS=0,1 bash remote/run_exp.sh exps/<实验名>.sh
```
- 实验在后台运行，关掉终端也不影响。看进度：`tail -f output/runs/<实验名>/RUN.log`
- `exps/*.sh` 是 Claude 写好的**自包含实验**：会自己做冒烟测试、训练（显存不够自动降 batch）、评测、盲测，并检查磁盘空间。
- 同一个实验中断后重跑，已完成的阶段会自动跳过。
- `GPUS` 指定用哪几张卡；有效 batch size 由脚本自动保持不变。

## 三、把结果带回给 Claude

实验结束时会**自动**打包：
```
output/bundles/<实验名>_<时间>.tar.gz     # 几 MB:日志(已裁剪)、指标、决策记录、帧条缩略图
```
同时在 `RUN.log` 末尾打印一份 `SUMMARY.md`。实验中途想看进度，可以随时手动打包：
```bash
bash remote/pack_results.sh <实验名>
```

带回方式任选其一：
1. **最省事**：复制 `SUMMARY.md` 的内容（几十行），直接贴进和 Claude 的对话。
2. **完整**：把 `.tar.gz` 传到开发机（从你能同时访问两边的电脑中转），然后在开发机上运行：
   ```bash
   bash tools/ingest_results.sh <结果包.tar.gz>     # 解压到 inbox/,Claude 可直接读
   ```

结果包**不含视频**。定性对比视频（`output/runs/<实验>/eval/*/`、盲测 `eval/blind/`）请在实验机上直接看，看完把结论告诉 Claude。

## 四、约定（Claude 写实验脚本时遵守）

- 所有路径都通过 `XN_*` 环境变量读取，代码里不写死机器路径（`src/utils/paths.py`）。
- 每个实验的所有产物都放在 `output/runs/<实验名>/` 下，便于打包。
- ckpt 只保留最近 4 个，启动前检查磁盘剩余空间。
- `meta.txt` 记录 git commit；如果工作区有未提交的改动，也会标出来，保证可复现。

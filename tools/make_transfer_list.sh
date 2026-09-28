#!/bin/bash
# 开发机:把实验机需要的东西按目标布局拷到 <DEST>(移动硬盘 / 中转位置)。两台机器网络不通,需人工搬运。
# 默认只搬**原始数据**(约 59G),帧 / latent / motion latent 在实验机上用 remote/prepare_data.sh 重新生成。
#
# 用法:
#   bash tools/make_transfer_list.sh --dry  <DEST>    # 只统计各组大小
#   bash tools/make_transfer_list.sh        <DEST>    # 实际拷贝(rsync,可断点续传,重复执行只补差)
#
# <DEST> 下的布局 → 实验机 configs/paths/<机器名>.env 里对应的变量:
#   hallo3_raw/            → XN_HALLO3_RAW   原始 mp4(只含定稿清单里的 clip)
#   hallo3/                → XN_HALLO3       定稿清单 *.txt + emo_pose_caption/(处理产物会生成在这里)
#   testset/               → XN_TESTSET      评测测试集(含 GT 快照,保证评测口径与开发机一致)
#   pretrained/            → XN_PRETRAINED
#   i3d_torchscript.pt     → XN_FVD_I3D
#   output/                → XN_OUTPUT       起点 ckpt(保持子目录名)
#   insightface_buffalo_l/ → 实验机 ~/.insightface/models/buffalo_l(抽帧的人脸检测模型)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; source "$REPO/configs/paths/ps.env"
DRY=0; [ "${1:-}" = "--dry" ] && { DRY=1; shift; }
DEST=${1:?用法: bash tools/make_transfer_list.sh [--dry] <DEST>}
H=$XN_HALLO3; RAW=/media/ps/ssd5/ayr/hallo3-data/videos
W=$REPO/output/transfer; mkdir -p "$W"
sed 's#$#.mp4#' "$H/manifest.txt" > "$W/mp4_list.txt"
sed 's#$#.json#' "$H/manifest.txt" > "$W/caption_list.txt"

# 组名 | 源 | 目标子路径 | files-from 清单(可空)
TGROUPS=(
  "原始 mp4|$RAW/|hallo3_raw/|$W/mp4_list.txt"
  "caption|$H/emo_pose_caption/|hallo3/emo_pose_caption/|$W/caption_list.txt"
  "测试集|$XN_TESTSET/|testset/|"
  "权重 sd-image-variations|$XN_PRETRAINED/sd-image-variations-diffusers/|pretrained/sd-image-variations-diffusers/|"
  "权重 SVD vae|$XN_PRETRAINED/stable-video-diffusion-img2vid/vae/|pretrained/stable-video-diffusion-img2vid/vae/|"
  "权重 xnemo_ckpt|$XN_PRETRAINED/xnemo_ckpt/|pretrained/xnemo_ckpt/|"
  "权重 umt5-base|$XN_PRETRAINED/umt5-base/|pretrained/umt5-base/|"
  "insightface|$HOME/.insightface/models/buffalo_l/|insightface_buffalo_l/|"
)
for f in manifest.txt train_data.txt train_data_ge64.txt testset_clips.txt overfit12.txt valid_data.txt; do
  TGROUPS+=("清单 $f|$H/$f|hallo3/$f|"); done
TGROUPS+=("FVD I3D|$XN_FVD_I3D|i3d_torchscript.pt|")
for f in s1_uni/stage1_step_750.pt s2_L64ft/stage2_step_2000.pt; do TGROUPS+=("ckpt $f|$XN_OUTPUT/$f|output/$f|"); done

total=0
for g in "${TGROUPS[@]}"; do IFS='|' read -r name src dst lst <<< "$g"
  if [ -n "$lst" ]; then
    sz=$(python3 -c "import os,sys;print(sum(os.path.getsize(os.path.join(sys.argv[1],l.strip())) for l in open(sys.argv[2]) if l.strip() and os.path.exists(os.path.join(sys.argv[1],l.strip()))))" "$src" "$lst")
  else sz=$(du -sb "$src" 2>/dev/null | cut -f1); fi
  total=$((total + ${sz:-0})); printf "%-34s %8.2f G\n" "$name" "$(echo "${sz:-0}/1000000000" | bc -l)"
  [ $DRY = 1 ] && continue
  mkdir -p "$(dirname "$DEST/$dst")"
  if [ -n "$lst" ]; then rsync -a --files-from="$lst" "$src" "$DEST/$dst"; else rsync -a "$src" "$DEST/$dst"; fi || echo "  ✗ $name 拷贝失败"
done
printf "%-34s %8.2f G\n" "== 合计" "$(echo "$total/1000000000" | bc -l)"
[ $DRY = 1 ] || echo "已拷到 $DEST。实验机上按文件头注释的对应关系填写 configs/paths/<机器名>.env。"

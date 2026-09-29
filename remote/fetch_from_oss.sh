#!/bin/bash
# 实验机:从阿里云 OSS 下载数据分卷 → 校验 → 解压。
# 用法: bash remote/fetch_from_oss.sh <下载目录> <解压目标根目录> [OSS 前缀]
# 前提:装好 ossutil 2.x 并配置凭证(见 docs/REMOTE_SETUP.md);bucket yanruan-avatar 在上海(cn-shanghai)。
#   实验机若在阿里云上海地域内网,可在 ~/.ossutilconfig 用 endpoint=https://oss-cn-shanghai-internal.aliyuncs.com(免流量费、更快)。
set -eu
DL=${1:?下载目录}; DST=${2:?解压目标根目录}
SRC=${3:-oss://yanruan-avatar/xnemo/transfer_20260928/}
OSSUTIL=${OSSUTIL:-ossutil}
mkdir -p "$DL"
echo "[1/2] 下载 $SRC → $DL(可断点续传,中断后重跑本命令即可)"
$OSSUTIL cp -r "$SRC" "$DL/" -j 4 --parallel 8 -u --checkpoint-dir "$DL/.oss_ckpt"
echo "[2/2] 校验并解压"
bash "$(dirname "$0")/unpack_data.sh" "$DL" "$DST"

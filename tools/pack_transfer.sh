#!/bin/bash
# 开发机:整理要搬的原始数据(约 61G)并切成 4G 分卷 + sha256,供网盘/中转电脑/移动硬盘任一方式传输。
# 用法: bash tools/pack_transfer.sh [STAGE=/media/ps/ssd5/ayr/xnemo_transfer]
#   产出 $STAGE/chunks/xnemo_data.tar.part_aa ... + SHA256SUMS + unpack_data.sh(实验机用)
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:-/media/ps/ssd5/ayr/xnemo_transfer}; mkdir -p "$STAGE/layout" "$STAGE/chunks"
echo "[1/3] 按目标布局整理 → $STAGE/layout"; bash "$REPO/tools/make_transfer_list.sh" "$STAGE/layout"
echo "[2/3] 打包并切成 4G 分卷"
( cd "$STAGE/layout" && tar -cf - . ) | split -b 4G -d -a 3 - "$STAGE/chunks/xnemo_data.tar.part_"
echo "[3/3] 校验和"; ( cd "$STAGE/chunks" && sha256sum xnemo_data.tar.part_* > SHA256SUMS )
cp "$REPO/remote/unpack_data.sh" "$STAGE/chunks/"
echo "完成:$(ls "$STAGE/chunks" | grep -c part_) 个分卷,共 $(du -sh "$STAGE/chunks" | cut -f1)。把 $STAGE/chunks/ 整个目录传到实验机即可。"

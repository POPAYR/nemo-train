#!/bin/bash
# 实验机:校验并解压数据分卷。用法: bash unpack_data.sh <分卷所在目录> <解压目标根目录>
#   目标根目录下会得到 hallo3_raw/ hallo3/ testset/ pretrained/ output/ i3d_torchscript.pt insightface_buffalo_l/
set -eu
SRC=${1:?分卷目录}; DST=${2:?解压目标根目录}; mkdir -p "$DST"
echo "[1/3] 校验 sha256(有损坏的分卷会列出来,重传那几个即可)"
( cd "$SRC" && sha256sum -c SHA256SUMS ) || { echo "✗ 校验失败,重传上面标 FAILED 的分卷后再运行"; exit 1; }
echo "[2/3] 解压 → $DST"; cat "$SRC"/xnemo_data.tar.part_* | tar -xf - -C "$DST"
echo "[3/3] insightface 模型 → ~/.insightface/models/buffalo_l"
mkdir -p ~/.insightface/models && cp -rn "$DST/insightface_buffalo_l" ~/.insightface/models/buffalo_l
D=$(cd "$DST" && pwd)
cat <<ENV

✓ 解压完成。把下面几行填进 configs/paths/\$(hostname -s).env(其余变量按 example.env 填):
export XN_HALLO3_RAW=$D/hallo3_raw
export XN_HALLO3=$D/hallo3
export XN_TESTSET=$D/testset
export XN_PRETRAINED=$D/pretrained
export XN_FVD_I3D=$D/i3d_torchscript.pt
export XN_OUTPUT=$D/output          # 起点 ckpt 已在这里;也可改到仓库内,再把 output/ 下的两个 ckpt 目录挪过去
ENV

#!/bin/bash
# 实验机:用 OSS 签名链接下载数据分卷(无需安装 ossutil、无需 AccessKey),可选直接校验并解压。
# 用法: bash remote/download_from_urls.sh <链接文件> <下载目录> [解压目标根目录]
#   链接文件:开发机生成的 download_urls.txt(每行一个签名 URL,7 天有效;相当于临时密码,勿外传、勿提交 git)
#   断点续传:中断后重跑同一命令,wget -c 会接着下。
set -eu
URLS=${1:?链接文件}; DL=${2:?下载目录}; DST=${3:-}
mkdir -p "$DL"; n=$(grep -c '^https' "$URLS"); i=0
while read -r u; do
  [ -z "$u" ] && continue; i=$((i+1))
  f=$(basename "${u%%\?*}")
  echo "[$i/$n] $f"
  wget -c -q --show-progress --tries=20 --retry-connrefused --waitretry=10 -O "$DL/$f" "$u" || { echo "✗ $f 下载失败(链接可能已过期,找开发机重新生成)"; exit 1; }
done < "$URLS"
echo "下载完成:$(ls "$DL" | wc -l) 个文件,$(du -sh "$DL" | cut -f1)"
if [ -n "$DST" ]; then bash "$DL/unpack_data.sh" "$DL" "$DST"; else echo "下一步:bash $DL/unpack_data.sh $DL <解压目标根目录>"; fi

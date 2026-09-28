#!/bin/bash
# 实验机数据处理:从原始 mp4 重新生成 hallo3 训练数据(帧 / 音频 / motion latent / VAE latent)。
# 用法: GPUS=0,1,2,3 bash remote/prepare_data.sh          (后台跑建议: setsid nohup ... &)
# 前提(随数据搬来,见 docs/REMOTE_SETUP.md):
#   $XN_HALLO3_RAW/<clip>.mp4
#   $XN_HALLO3/{manifest.txt, train_data.txt, train_data_ge64.txt, testset_clips.txt, overfit12.txt, valid_data.txt}
#   $XN_HALLO3/emo_pose_caption/
# 处理的是**开发机定稿的 clip 清单**(manifest.txt),不重跑背景运动过滤/清洗/测试集划分 ——
#   那些依赖阈值与人工标注,重跑可能得到不同的 clip 集合,实验结论就不能对照了。
# 与开发机一致的参数:fixed 裁剪、max_frames 256、512px、25fps;motion latent 用逐帧真实 bbox(--real_bbox 1)。
# 各阶段可重入:中断后重跑会跳过已完成的阶段与 clip。
set -u
source "$(dirname "$0")/env.sh" || exit 1
D=$XN_HALLO3; RAW=$XN_HALLO3_RAW; LIST=${LIST:-$D/manifest.txt}
RUN=$XN_OUTPUT/runs/_data_prep; LD=$RUN/logs; mkdir -p "$LD"
GPUS=${GPUS:-0}; G=(${GPUS//,/ })
N_EXT=${N_EXT:-$(( ${#G[@]} * 8 ))}; N_POSE=${N_POSE:-$(( ${#G[@]} * 2 ))}; N_LAT=${N_LAT:-$(( ${#G[@]} * 2 ))}
FACE_PY=${XN_FACE_PY:-$XN_PY}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RUN/RUN.log"; }
die(){ log "✗ 中止:$*"; exit 1; }
cnt(){ $XN_PY -c "import os,sys;d=sys.argv[1];print(sum(1 for x in os.listdir(d) if not x.startswith('.')) if os.path.isdir(d) else 0)" "$1"; }
cnt_dirs(){ $XN_PY -c "import os,sys;d=sys.argv[1];print(sum(1 for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))) if os.path.isdir(d) else 0)" "$1"; }
stage_done(){ [ -f "$RUN/.$1.done" ]; }
mark(){ touch "$RUN/.$1.done"; }
shards(){ # <名字> <分片数> <命令模板 {I} {N}>
  local nm=$1 n=$2 t=$3 pids=() i g c
  for i in $(seq 0 $((n-1))); do g=${G[$(( i % ${#G[@]} ))]}; c=${t//\{I\}/$i}; c=${c//\{N\}/$n}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash -c "$c" > "$LD/${nm}_$i.log" 2>&1 & pids+=($!); done
  log "  $nm:$n 个分片 @ GPU ${GPUS}"; for p in "${pids[@]}"; do wait $p; done; }
gate(){ # <名字> <实际> <期望> 低于 99% 就停
  local got=$2 want=$3; log "  $1:$got / $want"
  [ "$got" -ge $(( want * 99 / 100 )) ] || die "$1 产出不足 99%,查 $LD/"; }

TOTAL=$(grep -c . "$LIST"); log "═══ 数据处理开始:$TOTAL 条  GPUS=$GPUS ═══"
for f in manifest.txt train_data.txt train_data_ge64.txt testset_clips.txt; do [ -f "$D/$f" ] || die "缺 $D/$f(随数据搬来)"; done
[ -d "$RAW" ] || die "缺原始视频目录 $RAW"
NMP4=$($XN_PY -c "import os,sys;print(sum(os.path.exists(f'{sys.argv[1]}/{l.strip()}.mp4') for l in open(sys.argv[2]) if l.strip()))" "$RAW" "$LIST")
[ "$NMP4" -eq "$TOTAL" ] || die "原始 mp4 只有 $NMP4/$TOTAL 条"

if ! stage_done audio_raw; then log "▶ [1/7] 抽原始音频(CPU)"
  $XN_PY tools/data/extract_audio_raw.py "$LIST" --workers ${AUDIO_WORKERS:-32} >> "$LD/audio_raw.log" 2>&1 || die "抽音频失败,见 $LD/audio_raw.log"
  gate audio_wav_raw "$(cnt $D/audio_wav_raw)" $TOTAL; mark audio_raw; fi

if ! stage_done frames; then log "▶ [2/7] 抽帧(fixed 裁剪,insightface)"
  mkdir -p "$D/face_frames"
  shards ext $N_EXT "$FACE_PY tools/data/extract_frames_v2.py --list $LIST --src $RAW --shard {I}/{N} --mode fixed --out $D/face_frames --max_frames 256 --resume 1"
  gate face_frames "$(cnt_dirs $D/face_frames)" $TOTAL; mark frames; fi

if ! stage_done recut; then log "▶ [3/7] 按帧数切音频 → audio_pt / audio_wav"
  mkdir -p "$D/audio_pt" "$D/audio_wav"
  $XN_PY tools/data/recut_audio.py "$LIST" >> "$LD/recut.log" 2>&1 || die "切音频失败,见 $LD/recut.log"
  gate audio_pt "$(cnt $D/audio_pt)" $TOTAL; mark recut; fi

if ! stage_done pose; then log "▶ [4/7] motion latent(pose_embed_real)"
  mkdir -p "$D/pose_embed_real"
  shards pose $N_POSE "$XN_PY tools/data/extract_pose_embed_v2.py --frames $D/face_frames --out $D/pose_embed_real --list $LIST --shard {I}/{N} --real_bbox 1 --chunk 64 --resume 1"
  gate pose_embed_real "$(cnt $D/pose_embed_real)" $TOTAL; mark pose; fi

if ! stage_done latent; then log "▶ [5/7] VAE latent(frame_latent)"
  mkdir -p "$D/frame_latent"
  shards lat $N_LAT "$XN_PY tools/data/encode_latents.py --frames $D/face_frames --out $D/frame_latent --list $LIST --shard {I}/{N} --resume 1"
  gate frame_latent "$(cnt $D/frame_latent)" $TOTAL; mark latent; fi

log "▶ [6/7] 定稿校验(只校验,不改写清单)"
FC_VERIFY=1 $XN_PY tools/data/final_check.py "$LIST" 2>&1 | tee -a "$RUN/RUN.log" | tail -8
[ "${PIPESTATUS[0]}" -eq 0 ] || die "定稿校验未通过"

log "▶ [7/7] 数据指纹比对(与开发机)"
$XN_PY tools/data/data_fingerprint.py --ref docs/data_fingerprint_ref.json 2>&1 | grep -v pynvml | tee -a "$RUN/RUN.log" | tail -4
[ "${PIPESTATUS[0]}" -eq 0 ] || die "数据指纹与开发机不一致(见上);先别训练,把 RUN.log 发给 Claude"
log "═══ 数据就绪 ═══  占用 $(du -sh "$D" 2>/dev/null | cut -f1)"
echo "[DATA_READY]" >> "$RUN/RUN.log"

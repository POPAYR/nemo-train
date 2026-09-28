#!/bin/bash
# 实验:64 帧微调的恒定 lr 续训 → 退火 → 20 clip 评测(与旧 64 微调同口径对比)+ 眼/嘴盲测
# 问题:s2_L64ft 的 vmse 末段变平,是真收敛还是 lr 余弦退火到 1e-6 把权重"冻住"了?
#   判据:恒定 lr 阶段 vmse 能否明显低于 0.16794(s2_L64ft@2000,8 clip 分片 val)。
# 起点:output/s2_L64ft/stage2_step_2000.pt(EMA)+ output/s1_uni/stage1_step_750.pt
# 用法(实验机):GPUS=0,1 bash remote/run_exp.sh exps/20260928_L64ft2.sh
set -u
source "$(dirname "$0")/../remote/env.sh" || exit 1
EXP=${EXP:-20260928_L64ft2}; RUN=${RUN:-$XN_OUTPUT/runs/$EXP}; mkdir -p "$RUN/logs"
# 规模参数(DRY 测试时调小;正式实验用默认值)
STEPS_A=${STEPS_A:-2000}; STEPS_B=${STEPS_B:-1000}; SAVE=${SAVE:-500}; VALE=${VALE:-250}; VALS=${VALS:-500}; NCLIP=${NCLIP:-20}
GPUS=${GPUS:-0,1}; NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
S1=$XN_OUTPUT/s1_uni/stage1_step_750.pt
START=$XN_OUTPUT/s2_L64ft/stage2_step_2000.pt
log(){ echo "[$(date '+%m-%d %H:%M')] $*"; }
need_disk(){ local g=$(df -BG --output=avail "$XN_OUTPUT" | tail -1 | tr -dc 0-9); [ "$g" -ge "$1" ] || { log "✗ 磁盘仅剩 ${g}G(需 ≥$1G),中止"; exit 1; }; }

[ -f "$S1" ] && [ -f "$START" ] || { log "✗ 缺起点 ckpt:$S1 或 $START(见 docs/REMOTE_SETUP.md 数据清单)"; exit 1; }
log "开始 $EXP  GPUS=$GPUS(${NG} 卡)  git=$(git rev-parse --short HEAD)"
need_disk 60

COMMON="--rope --stage1_ckpt $S1 --resume_use_ema --pose_real --bbox_drop 0 --sources 0 --grad_stat --t_mode uniform
 --L 64 --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --ema_device auto --val_cfg 2.0 --val_window 0
 --save_every $SAVE --keep_last 4 --log_every 20 --val_every $VALE --val_sample_every $VALS --val_clips 8 --val_sample_steps 20"
# eff_bsz 固定 8:NG 卡 × batch × accum
ACC2=$(( 8 / (NG * 2) )); ACC1=$(( 8 / NG ))

train(){ # $1=阶段名 $2=resume ckpt $3..=额外参数
  local ph=$1 ck=$2; shift 2; local out=$RUN/$ph lg=$RUN/logs/$ph.log
  if grep -q "^\[done\]" "$lg" 2>/dev/null; then log "$ph 已完成,跳过"; return 0; fi
  need_disk 40
  for cfg in "2 $ACC2" "1 $ACC1"; do set -- $cfg "$@"; local b=$1 a=$2; shift 2
    log "$ph 启动:batch$b × accum$a × ${NG}卡"
    CUDA_VISIBLE_DEVICES=$GPUS $XN_PY -u -m torch.distributed.run --nproc_per_node=$NG --master_port $((29000 + RANDOM % 900)) \
      scripts/train/flow_stage2_temporal.py $COMMON --batch $b --accum $a --resume "$ck" --out "$out" "$@" > "$lg" 2>&1
    if grep -q "^\[done\]" "$lg"; then
      log "$ph ✓ 完成;vmse:$(grep -E '^\[val\] step' "$lg" | awk '{printf "%s:%s ", $3, substr($4,11)}')"; return 0; fi
    if grep -q "out of memory" "$lg" && [ "$b" = 2 ]; then log "$ph batch2 OOM → 退到 batch1"; mv "$lg" "$lg.oom_b2"; continue; fi
    log "✗ $ph 失败:$(grep -m1 -E 'Error|out of memory' "$lg" | cut -c1-160)"; exit 1
  done
}

# 冒烟(CLAUDE.md §6.8)
SM=$RUN/smoke; if ! grep -q "^\[done\]" "$SM/smoke.log" 2>/dev/null; then mkdir -p "$SM"
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} $XN_PY -u scripts/train/flow_stage2_temporal.py --smoke --rope --stage1_ckpt "$S1" \
    --resume "$START" --resume_use_ema --pose_real --bbox_drop 0 --sources 0 --t_mode uniform --lr 1.5e-5 \
    --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --val_every 0 --out "$SM" > "$SM/smoke.log" 2>&1
  grep -q "^\[done\]" "$SM/smoke.log" || { log "✗ 冒烟失败,见 $SM/smoke.log"; exit 1; }
  log "冒烟通过:$(grep '^step' "$SM/smoke.log" | tail -1 | cut -c1-80)"; fi

# 阶段 A:恒定 lr 1.5e-5,2000 步;阶段 B:余弦 1.5e-5 → 1e-6,1000 步
train A_const "$START"                           --lr 1.5e-5 --max_steps $STEPS_A
train B_anneal "$RUN/A_const/stage2_step_$STEPS_A.pt" --lr 1.5e-5 --lr_anneal_steps $STEPS_B --lr_min 1e-6

# 评测:20 clip × 64 帧,单窗,新(B@1000)vs 旧(s2_L64ft@2000),同 seed 同 clip
EV=$RUN/eval; L=$EV/_lists; mkdir -p "$L"
$XN_PY - "$L" "$NCLIP" <<'PY'
import json, sys
from src.utils.paths import P as XP
man = json.load(open(XP("XN_TESTSET", "manifest.json")))["hallo3"]
sel = set(json.load(open(XP("XN_TESTSET", "hallo3_subset30.json"))))
n = int(sys.argv[2]); c = sorted(x for x in man if x in sel)[:n]
h = n // 2; open(sys.argv[1] + "/a.txt", "w").write("\n".join(c[:h])); open(sys.argv[1] + "/b.txt", "w").write("\n".join(c[h:]))
PY
G1=${GPUS%%,*}; G2=${GPUS##*,}
render(){ # $1=输出名 $2=ckpt
  mkdir -p "$EV/$1"
  for part in "a $G1 1234" "b $G2 $((1234 + NCLIP / 2))"; do set -- "$1" "$2" $part
    CUDA_VISIBLE_DEVICES=$4 VAL_CLIPS_FILE=$L/$3.txt $XN_PY -u tools/render_s2_cfg.py --s2_ckpt "$2" --stage1_ckpt "$S1" --use_ema \
      --window 0 --shift 1.0 --cfgs 2.0 --steps 20 --clips $((NCLIP / 2)) --data_seed $5 --out "$EV/$1" > "$EV/$1/render_$3.log" 2>&1 &
  done; wait
  log "渲染 $1:$(ls "$EV/$1"/*_cfg2.0.mp4 2>/dev/null | wc -l) 条"; }
render new_B1000 "$RUN/B_anneal/stage2_step_$STEPS_B.pt"
render new_A2000 "$RUN/A_const/stage2_step_$STEPS_A.pt"
render old_L64ft "$START"
NM=("新:续训+退火" "新:恒定lr@2000" "旧:64微调@2000")
CUDA_VISIBLE_DEVICES=$G1 $XN_PY tools/video_metrics.py --dirs "$EV/new_B1000" "$EV/new_A2000" "$EV/old_L64ft" --names "${NM[@]}" --work "$EV/_fid_png" > "$EV/metrics.log" 2>&1
CUDA_VISIBLE_DEVICES=$G1 $XN_PY tools/teacher_compare.py --dirs "$EV/new_B1000" "$EV/new_A2000" "$EV/old_L64ft" --names "${NM[@]}" > "$EV/regions.log" 2>&1
[ -n "${XN_FACE_PY:-}" ] && $XN_FACE_PY tools/region_zoom_blind.py --dirs "$EV/new_B1000" "$EV/old_L64ft" --names "新:续训+退火" "旧:64微调" --seed 11 --out "$EV/blind" > "$EV/blind.log" 2>&1
log "评测完成:$EV/metrics.log  $EV/regions.log  盲测 $EV/blind/"
log "[EXP_DONE]"

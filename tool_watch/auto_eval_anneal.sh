#!/bin/bash
# 自主编排(评测支线):等 auto_L64_ft 选出退火最优 ckpt → 20 clip 渲染(官方滑窗)→ 指标 → 眼/嘴盲测
cd /media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python; FPY=/home/ayr/miniconda3/envs/face/bin/python
W=output/_logs/active/auto_L64_ft.watch; T=output/eval/teacher_cmp
log(){ echo "[$(date +%m-%d\ %H:%M)] [eval] $*" >> $W; }
until [ -s output/_logs/active/auto_L64_ft.best_ckpt ]; do sleep 60; done
CK=$(cat output/_logs/active/auto_L64_ft.best_ckpt); ST=$(basename $CK .pt | sed 's/stage2_step_//')
O=$T/L24anneal_s$ST; mkdir -p $O
log "开始评测退火最优 $CK → $O"
freegpu(){ # 选 1 张剩余显存 ≥ 26GB 的卡,跳过训练用的 2,7
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=2 && $1!=7 && $2<55000 {print $1}' | grep -vx "$1" | head -1)
    [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
G1=$(freegpu); G2=$(freegpu $G1); log "渲染用 GPU $G1,$G2"
for part in a:$G1:1234 b:$G2:1244; do IFS=: read L G S <<< "$part"
  env CUDA_VISIBLE_DEVICES=$G VAL_CLIPS_FILE=$T/L24c_s9000/_lists/$L.txt $PY -u /media/ps/ssd5/ayr/tool/render_s2_cfg.py \
    --s2_ckpt $CK --stage1_ckpt output/s1_uni/stage1_step_750.pt --use_ema \
    --window 24 --overlap 4 --ctx official --shift 1.0 --cfgs 2.0 --steps 20 --clips 10 --data_seed $S \
    --out $O > $O/render_$L.log 2>&1 &
done; wait
until grep -q "\[done\]" $T/L24c_s12000/render_a.log && grep -q "\[done\]" $T/L24c_s12000/render_b.log; do sleep 60; done
n=$(ls $O/*_cfg2.0.mp4 | wc -l); log "渲染完成 $n 条"
D="$O $T/L24c_s12000 $T/L24c_s9000 $T/flow $T/eps"
NM=("24退火@$ST" "24退火前@12000" "24@9000" "64_S单窗" "eps(cfg2.5)")
CUDA_VISIBLE_DEVICES=$G1 $PY /media/ps/ssd5/ayr/tool/video_metrics.py --dirs $D --names "${NM[@]}" --work $T/_metrics_anneal > $O/metrics.log 2>&1
CUDA_VISIBLE_DEVICES=$G1 $PY /media/ps/ssd5/ayr/tool/teacher_compare.py --dirs $D --names "${NM[@]}" > $O/regions.log 2>&1
log "指标完成:$O/metrics.log  $O/regions.log"
B=output/eval/blind_anneal_s$ST
$FPY /media/ps/ssd5/ayr/tool/region_zoom_blind.py --dirs $O $T/L24c_s12000 --names "退火后@$ST" "退火前@12000" --out $B > $B.log 2>&1
log "盲测视频完成:$B(答案在 $B/_key/key.json)"
log "[EVAL_DONE]"

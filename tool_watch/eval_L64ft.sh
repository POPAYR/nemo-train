#!/bin/bash
cd /media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python; FPY=/home/ayr/miniconda3/envs/face/bin/python
T=output/eval/teacher_cmp; L=$T/L24c_s9000/_lists; W=output/_logs/active/auto_L64_ft.watch
log(){ echo "[$(date +%m-%d\ %H:%M)] [eval2] $*" >> $W; }
A=$T/L64ft_s2000; B=$T/L24anneal_s4000; mkdir -p $A $B
log "补评测:L64ft/step_2000(单窗64) 与 anneal/step_4000(滑窗24)"
R(){ # out ckpt window list gpu seed
  env CUDA_VISIBLE_DEVICES=$5 VAL_CLIPS_FILE=$L/$4.txt $PY -u /media/ps/ssd5/ayr/tool/render_s2_cfg.py \
    --s2_ckpt $2 --stage1_ckpt output/s1_uni/stage1_step_750.pt --use_ema \
    --window $3 --overlap 4 --ctx official --shift 1.0 --cfgs 2.0 --steps 20 --clips 10 --data_seed $6 \
    --out $1 > $1/render_$4.log 2>&1; }
R $A output/s2_L64ft/stage2_step_2000.pt 0 a 6 1234 &
R $A output/s2_L64ft/stage2_step_2000.pt 0 b 2 1244 &
R $B output/s2_L24c_anneal/stage2_step_4000.pt 24 a 7 1234 &
R $B output/s2_L24c_anneal/stage2_step_4000.pt 24 b 6 1244 &
wait; log "渲染完成:L64ft $(ls $A/*_cfg2.0.mp4|wc -l) 条,anneal4000 $(ls $B/*_cfg2.0.mp4|wc -l) 条"
D="$A $B $T/L24anneal_s1000 $T/L24c_s12000 $T/flow $T/eps"
NM=("64微调@2000(单窗)" "24退火@4000(滑窗)" "24退火@1000(滑窗)" "24退火前@12000" "64_S单窗" "eps(cfg2.5)")
CUDA_VISIBLE_DEVICES=6 $PY /media/ps/ssd5/ayr/tool/video_metrics.py --dirs $D --names "${NM[@]}" --work $T/_metrics_L64ft > $A/metrics.log 2>&1
CUDA_VISIBLE_DEVICES=6 $PY /media/ps/ssd5/ayr/tool/teacher_compare.py --dirs $D --names "${NM[@]}" > $A/regions.log 2>&1
log "指标完成:$A/metrics.log $A/regions.log"
Bl=output/eval/blind_L64ft_vs_S_vs_24
$FPY /media/ps/ssd5/ayr/tool/region_zoom_blind.py --dirs $A $T/flow $B --names "64微调@2000" "64_S单窗" "24退火@4000" --seed 7 --out $Bl > $Bl.log 2>&1
log "盲测完成:$Bl(答案 $Bl/_key/key.json)"
log "[EVAL2_DONE]"

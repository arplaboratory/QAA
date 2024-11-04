# Baseline
export CONFIG=configs/eval/eval_sfxl.yaml CKPT=cliquemining.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval_salad.sbatch

# Ours
export CONFIG=configs/eval/eval_sfxl.yaml CKPT=best.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval_salad.sbatch

# BoQ
export CONFIG=configs/eval/eval_sfxl.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/eval/eval_boq.sbatch

# VIS
export CONFIG=configs/eval/eval_vis.yaml CKPT=best.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval_salad_vis.sbatch
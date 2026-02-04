# Baseline
export CONFIG=configs/eval/eval.yaml CKPT=cliquemining.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval.sbatch

# Ours
export CONFIG=configs/eval/eval.yaml CKPT=best/best_2025_8192.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval.sbatch

# BoQ
export CONFIG=configs/eval/eval.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/eval/eval_boq.sbatch

# Baseline SFXL
export CONFIG=configs/eval/eval_sfxl.yaml CKPT=cliquemining.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval.sbatch

# Ours SFXL
export CONFIG=configs/eval/eval_sfxl.yaml CKPT=best/best_2025_8192.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval.sbatch

# BoQ SFXL
export CONFIG=configs/eval/eval_sfxl.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/eval/eval_boq.sbatch

# VIS
export CONFIG=configs/eval/eval_vis.yaml CKPT=best.ckpt
sbatch --export=ALL,CONFIG=$CONFIG,CKPT=$CKPT scripts/eval/eval_vis.sbatch
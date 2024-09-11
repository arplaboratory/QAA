export CONFIG=configs/train/baseline/train_salad_gsv_msls_topk_ori.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/baseline/train_salad_gsv_topk_ori.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/baseline/train_salad_msls_topk_ori.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_gsv_msls_topk_ori.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_topk_ori.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch

export CONFIG=configs/train/SALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch

export CONFIG=configs/train/QSALAD/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch

export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b2_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b6_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
export CONFIG=configs/train/QSALAD_recompute/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b6.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_long.sbatch
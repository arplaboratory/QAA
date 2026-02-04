# Multi-dataset joint training - SALAD Baseline
export CONFIG=configs/train/baseline/train_salad_gsv_topk_ori_b2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_msls_topk_ori_b2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_topk_ori_b2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_gsv_msls_topk_ori_b2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_gsv_msls_topk_ori_s2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# Multi-dataset joint training - QAA
export CONFIG=configs/train/agg/train_salad_gsv_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_msls_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_s2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# learned query vs conditioned
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_ot_cond.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_ot.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_soft_cond.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_soft.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# query scalability conditioned
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_16_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_32_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# reduced dim
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_8_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_16_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_16_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_32_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_16_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_32_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
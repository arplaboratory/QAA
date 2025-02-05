# Multi-dataset naive joint training
export CONFIG=configs/train/baseline/train_salad_gsv_msls_topk_ori_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_gsv_topk_ori_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_msls_topk_ori_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/baseline/train_salad_sf_xl_topk_ori_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_b2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_b2_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# SALAD Dim
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_64.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_32.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_16.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# BoQ Dim
export CONFIG=configs/train/agg_recompute_boq/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_boq_64.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_boq/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_boq_32.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_boq/train_salad_sf_xl_gsv_msls_topk_ori_b2_re_boq_16.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# learned query vs conditioned
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_con.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# query scalability conditioned
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_16_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_32_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_64_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_128_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_256_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_con/train_salad_sf_xl_gsv_msls_topk_ori_512_16_512_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_16_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_32_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_64_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_128_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_512_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# Reduced dim
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_16_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_32_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# freeze
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_backbone.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_feature.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_score.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# NH
export CONFIG=configs/train/agg_recompute_nh/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_4.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_nh/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_8.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_nh/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re_12.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
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

# learned query vs conditioned
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_con.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_con_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re.yaml
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

# different nq and cs
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_16_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_32_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_64_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_128_16_512_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_32_32_256_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_64_32_256_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_128_32_256_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_64_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_64_128_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_128_128_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_256_128_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_dim/train_salad_sf_xl_gsv_msls_topk_ori_512_128_64_b2_de_re.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# freeze
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_backbone.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_feature.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_freeze/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_score.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# self attention
export CONFIG=configs/train/agg_recompute_sa/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_sa_feature.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_sa/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_sa_none.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/agg_recompute_sa/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_re_sa_score.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch

# final
export CONFIG=configs/train/final3/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_lr2_long.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
export CONFIG=configs/train/final3/train_salad_sf_xl_gsv_msls_topk_ori_128_64_128_b2_de_lr2.yaml
sbatch --export=ALL,CONFIG=$CONFIG scripts/train/train_salad_longer.sbatch
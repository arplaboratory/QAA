#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=24               # uses 1 compute core per task
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=pr_122_tandon_advanced
#SBATCH --mem=350GB
#SBATCH --job-name=eval_salad
#SBATCH --output=eval_salad.out

overlay_path_gsv="datasets_sqf/GSV.sqf"
overlay_path_pitts30k="datasets_sqf/pitts30k.sqf"
overlay_path_pitts250k="datasets_sqf/pitts250k.sqf"
overlay_path_msls="datasets_sqf/mapillary_sls.sqf"
overlay_path_svox="datasets_sqf/svox.sqf"
overlay_path_nordland="datasets_sqf/nordland.sqf"
overlay_path_nordland_subset="datasets_sqf/nordland_subset.sqf"
overlay_path_sped="datasets_sqf/sped.sqf"
overlay_path_tokyo247="datasets_sqf/tokyo247.sqf"
overlay_path_eynsham="datasets_sqf/eynsham.sqf"
overlay_path_amstertime="datasets_sqf/amstertime.sqf"
overlay_path_SF_XL_val="datasets_sqf/SF_XL_val.sqf"
overlay_path_SF_XL_test="datasets_sqf/SF_XL_test.sqf"
overlay_path_SF_XL_train="datasets_sqf/SF_XL_train.sqf"

singularity exec --nv \
                --bind /etc/pki/ca-trust/extracted/pem/ \
                --overlay $overlay_path_gsv:ro \
                --overlay $overlay_path_pitts30k:ro \
                --overlay $overlay_path_pitts250k:ro \
                --overlay $overlay_path_msls:ro \
                --overlay $overlay_path_svox:ro \
                --overlay $overlay_path_nordland:ro \
                --overlay $overlay_path_nordland_subset:ro \
                --overlay $overlay_path_sped:ro \
                --overlay $overlay_path_tokyo247:ro \
                --overlay $overlay_path_eynsham:ro \
                --overlay $overlay_path_amstertime:ro \
                --overlay $overlay_path_SF_XL_val:ro \
                --overlay $overlay_path_SF_XL_test:ro \
                --overlay $overlay_path_SF_XL_train:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVPR; python3 -u eval.py --config $CONFIG --ckpt_path $CKPT --visualize"
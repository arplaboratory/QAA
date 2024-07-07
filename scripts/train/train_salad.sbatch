#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=12               # uses 1 compute core per task
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --job-name=train_salad_gsv
#SBATCH --output=train_salad_gsv.out

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
overlay_path_svox_night="datasets_sqf/svox_night.sqf"
overlay_path_svox_overcast="datasets_sqf/svox_overcast.sqf"
overlay_path_svox_rain="datasets_sqf/svox_rain.sqf"
overlay_path_svox_snow="datasets_sqf/svox_snow.sqf"
overlay_path_svox_sun="datasets_sqf/svox_sun.sqf"

singularity exec --nv \
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
                --overlay $overlay_path_svox_night:ro \
                --overlay $overlay_path_svox_overcast:ro \
                --overlay $overlay_path_svox_rain:ro \
                --overlay $overlay_path_svox_snow:ro \
                --overlay $overlay_path_svox_sun:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 -u main.py --config configs/train/"$CONFIG".yaml"
#!/bin/bash 

overlay_path="datasets_sqf/sped.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name SPED"

overlay_path="datasets_sqf/eynsham.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name eynsham"

overlay_path="datasets_sqf/amstertime.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name amstertime"

overlay_path="datasets_sqf/tokyo247.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name tokyo247"

overlay_path="datasets_sqf/pitts30k.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts30k --split train; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts30k --split val; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts30k --split test;"

overlay_path="datasets_sqf/pitts250k.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts250k --split train; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts250k --split val; python3 dataloaders/GenerateDatasetNpy.py --dataset_name pitts250k --split test;"

overlay_path="datasets_sqf/nordland.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name nordland"

overlay_path="datasets_sqf/nordland_subset.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name nordland_subset"

overlay_path="datasets_sqf/svox.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVG; python3 dataloaders/GenerateDatasetNpy.py --dataset_name svox --split train; python3 dataloaders/GenerateDatasetNpy.py --dataset_name svox --split val; python3 dataloaders/GenerateDatasetNpy.py --dataset_name svox --split test"
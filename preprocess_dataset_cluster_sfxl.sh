#!/bin/bash 

overlay_path="datasets_sqf/SF_XL_train.sqf"
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate UniVPR; python3 dataloaders/GenerateDatasetCluster.py --dataset_name SF_XL"
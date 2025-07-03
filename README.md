# Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition

This repository is the official implementation for [Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition]().

## Summary

We introduce Query-based Adaptive Aggregation (QAA) to expand the model memory capacity, leading to better generalization performance for diverse datasets. We also introduce the QAA framework for efficient multi-dataset joint training.

## Setup

Create a conda environment with the following:
```
conda env create -f environment.yml
```

## Dataset

For training, download [GSV-Cities](https://github.com/amaralibey/gsv-cities), [MSLS](https://www.mapillary.com/dataset/places), and [SF-XL](https://docs.google.com/forms/d/e/1FAIpQLSdQEcRULPLNr0Zk5x85jNw3vcel_RxoQoKtsrJA7QPjWPVqZg/viewform). 

For evaluation, download and format the desired datasets from [VPR-dataset-downloader](https://github.com/gmberton/VPR-datasets-downloader/tree/main), except for [Nordland*](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W) and MSLS (using official dataset).

### Option 1: Compress datasets into sqf files
For the best compatibility, compress dataset folders into a single `.sqf` file using `mksquashfs`. Example for MSLS:
```
mksquashfs mapillary_sls mapillary_sls.sqf  -keep-as-directory
```
Place the resulting `.sqf` file in the `datasets_sqf` directory.

### Option 2: Utilize original datasets
If you don't want to use `.sqf` files, just put the dataset folder into `datasets` folder.

## Preprocess

### Option 1: Preprocess from Scratch
Run the following scripts:

```
./preprocess_dataset_npy.sh
./preprocess_dataset_cluster_sfxl.sh  # For clustering the SF-XL training set
```

The preprocessing results will be saved in the cache/datasets directory. The provided code relies on [singularity](https://github.com/sylabs/singularity). If you do not have singularity installed, replace the following example command:
```
singularity exec --overlay $overlay_path:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate QAA; python3 dataloaders/GenerateDatasetNpy.py --dataset_name SPED"
```
with:
```
mount $overlay_path / -t squashfs -o loop # Do this only if you use .sqf files; Otherwise skip mounting
source ~/.bashrc; conda activate QAA; python3 dataloaders/GenerateDatasetNpy.py --dataset_name SPED
```

To include a new dataset, refer to the scripts `preprocess_dataset_npy.sh` and `dataloaders/GenerateDatasetNpy.py` for instructions on generating `.npy` files.

### Option 2: Use Preprocessed Data
Download preprocessed dataset caches from [link]() (released after review). Place the files in the `cache/datasets` directory.

## Train

All training scripts are included in `train.sh`, with config files in `configs/train` folder. The provided code relies on [singularity](https://github.com/sylabs/singularity). If you do not have singularity installed, replace the `scripts/train/train_salad_longer.sbatch`:
```
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
                --overlay $overlay_path_SF_XL_val:ro \
                --overlay $overlay_path_SF_XL_test:ro \
                --overlay $overlay_path_SF_XL_train:ro \
                 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate QAA; python3 -u main.py --config $CONFIG"
```
with:
```
# Do this only if you use .sqf files; Otherwise skip mounting
mount $overlay_path_gsv / -t squashfs -o loop
mount $overlay_path_pitts30k / -t squashfs -o loop
mount $overlay_path_pitts250k / -t squashfs -o loop
mount $overlay_path_msls / -t squashfs -o loop
mount $overlay_path_svox / -t squashfs -o loop
mount $overlay_path_nordland / -t squashfs -o loop
mount $overlay_path_nordland_subset / -t squashfs -o loop
mount $overlay_path_sped / -t squashfs -o loop
mount $overlay_path_tokyo247 / -t squashfs -o loop
mount $overlay_path_eynsham / -t squashfs -o loop
mount $overlay_path_amstertime / -t squashfs -o loop
mount $overlay_path_SF_XL_val / -t squashfs -o loop
mount $overlay_path_SF_XL_test / -t squashfs -o loop
mount $overlay_path_SF_XL_train / -t squashfs -o loop
source ~/.bashrc; conda activate QAA; python3 -u main.py --config $CONFIG
```

## Evaluation

All evaluation scripts are included in `eval.sh`, with config files in `configs/eval` folder. The provided code relies on [singularity](https://github.com/sylabs/singularity). If you do not have singularity installed, please replace the code in `scripts/eval` similar to the training script.

## Model weights

The model weights are provided in [link]() (released after review).

## Acknowledgements
This code is based on the amazing work of:
 - [CliqueMining](https://github.com/serizba/cliquemining)
 - [BoQ](https://github.com/amaralibey/Bag-of-Queries)
 - [MixVPR](https://github.com/amaralibey/MixVPR)
 - [GSV-Cities](https://github.com/amaralibey/gsv-cities)
 - [DINOv2](https://github.com/facebookresearch/dinov2)

## Cite
Here is the bibtex to cite our paper
```
TBD
```

# Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition

This repository is the official implementation for [Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition]().

## Summary

We introduce Query-based Adaptive Aggregation (QAA) to expand the model memory capacity, leading to better generalization performance for diverse datasets. We also introduce the UniVPR framework for efficient multi-dataset joint training.

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

The results will be stored in the `cache/datasets` directory.

To add a new dataset, refer to `dataloaders/GenerateDatasetNpy.py` for guidance on generating .npy files.

### Option 2: Use Preprocessed Data
Download preprocessed dataset caches from [link]() (released after review). Place the files in the `cache/datasets` directory.

## Train

The training script is `train.sh`, with config files in `configs/train` folder.

## Evaluation

You can download a pretrained UniVPR model from [here]() (released after review). For evaluating run:

```bash
python3 eval.py --ckpt_path 'weights/dino_salad.ckpt' --image_size 322 322 --batch_size 256 --val_datasets MSLS Nordland
```

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

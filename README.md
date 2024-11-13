# Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition

This repository is the official implementation for [Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition]().

## Summary

We introduce Query-based Adaptive Aggregation (QAA) to expand the model memory capacity, leading to better generalization performance for diverse datasets. We also introduce the UniVPR framework for efficient multi-dataset joint training.

## Setup

Create a conda environment with the following:
```
conda env create -f environment.yml
```

To quickly test and use our model, you can use Torch Hub:
```python
import torch
model = torch.hub.load("serizba/salad", "dinov2_salad")
model.eval()
model.cuda()
```

## Dataset

For training, download [GSV-Cities](https://github.com/amaralibey/gsv-cities), [MSLS](https://www.mapillary.com/dataset/places), and [SF-XL](https://docs.google.com/forms/d/e/1FAIpQLSdQEcRULPLNr0Zk5x85jNw3vcel_RxoQoKtsrJA7QPjWPVqZg/viewform). For evaluation, download and format the desired datasets from [VPR-dataset-downloader](https://github.com/gmberton/VPR-datasets-downloader/tree/main), except for [Nordland*](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W) and MSLS (using official dataset).

## Train

Training is done on GSV-Cities for 4 complete epochs. It requires around 30 minutes on an NVIDIA RTX 3090. For training DINOv2 SALAD run:
```bash
python3 main.py
```

After training, logs and checkpoints should be on the `logs` dir.

## Evaluation

You can download a pretrained DINOv2 SALAD model from [here](https://drive.google.com/file/d/1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1/view?usp=sharing). For evaluating run:

```bash
python3 eval.py --ckpt_path 'weights/dino_salad.ckpt' --image_size 322 322 --batch_size 256 --val_datasets MSLS Nordland
```

<table>
<thead>
  <tr>
    <th colspan="3">MSLS Challenge</th>
    <th colspan="3">MSLS Val</th>
    <th colspan="3">NordLand</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>75.0</td>
    <td>88.8</td>
    <td>91.3</td>
    <td>92.2</td>
    <td>96.4</td>
    <td>97.0</td>
    <td>76.0</td>
    <td>89.2</td>
    <td>92.0</td>
  </tr>
</tbody>
</table>

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

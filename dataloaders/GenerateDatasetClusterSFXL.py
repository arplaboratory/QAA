import os
import numpy as np
from glob import glob
from os.path import join
from sklearn.neighbors import NearestNeighbors
import argparse
from mapillary_sls.mapillary_sls.datasets.msls import MSLS
import shutil
import pandas as pd
from GenerateDatasetUtils import initialize

NPY_ROOT = 'cache/datasets/'

def generate_dataset_cluster_groups(datasets_folder, dataset_name, val_positive_dist_threshold=25.0):
    #### For SF_XL
    classes_per_group, images_per_class = initialize(dataset_folder = os.path.join(datasets_folder, dataset_name, "train"))
    np.save("cache/datasets/SF_XL/clusters.npy", images_per_class, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset in .npy format")
    parser.add_argument("--datasets_folder", type=str, default="datasets", help="Folder containing the datasets")
    parser.add_argument("--dataset_name", type=str, default="SF_XL", help="Name of the dataset")
    args = parser.parse_args()
    if args.dataset_name == "SF_XL":
        generate_dataset_cluster_groups(args.datasets_folder, args.dataset_name)
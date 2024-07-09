import os
import numpy as np
from glob import glob
from os.path import join
from sklearn.neighbors import NearestNeighbors
import argparse
from mapillary_sls.mapillary_sls.datasets.msls import MSLS
import shutil

def generate_dataset_npy(datasets_folder, dataset_name, split, val_positive_dist_threshold=25):
    #### Read paths and UTM coordinates for all images.
    if args.dataset_name == "svox":
        dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        svox_query_types = ["queries", "queries_night", "queries_overcast", "queries_rain", "queries_snow", "queries_sun"]
        if split=="val":
            database_folder = join(dataset_folder, "gallery")
            queries_folder = join(dataset_folder, "queries")
            save_dataset_npy(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)
        else:
            database_folder = join(dataset_folder, "gallery")
            for query_type in svox_query_types:
                queries_folder = join(dataset_folder, query_type)
                if query_type != "queries":
                    save_dataset_npy(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold, suffix="_" + query_type.split("_")[-1])
                else:
                    save_dataset_npy(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)
    elif args.dataset_name == "SF_XL":
        dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        sfxl_query_types = ["queries_night", "queries_occlusion", "queries_v1", "queries_v2"]
        database_folder = join(dataset_folder, "database")
        if split=="train": # Only DB images
            queries_folder = None
            save_dataset_npy_sfxl(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)
        if split=="val":
            queries_folder = join(dataset_folder, "queries")
            save_dataset_npy_sfxl(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)
        elif split=="test":
            for query_type in sfxl_query_types:
                queries_folder = join(dataset_folder, query_type)
                if query_type != "queries":
                    save_dataset_npy_sfxl(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold, suffix="_" + query_type.split("_")[-1])
                else:
                    save_dataset_npy_sfxl(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)
    elif args.dataset_name == "mapillary_sls":
        dataset_folder = join(datasets_folder, dataset_name)
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        if os.path.isdir("npys"):
            shutil.rmtree("npys")
        os.mkdir("npys")
        if split == "val":
            _ = MSLS(root_dir=dataset_folder, save=True, mode=split, posDistThr=25)
        else:
            _ = MSLS(root_dir=dataset_folder, save=True, mode=split)
        
        shutil.copytree("npys", os.path.join(f"cache/datasets/{dataset_name}/"), dirs_exist_ok=True)
        shutil.rmtree("npys")
    else:
        dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
        database_folder = join(dataset_folder, "database")
        queries_folder = join(dataset_folder, "queries")
        save_dataset_npy(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold)

def save_dataset_npy(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold, suffix=""):
    if not os.path.exists(database_folder):
        raise FileNotFoundError(f"Folder {database_folder} does not exist")
    if not os.path.exists(queries_folder):
        raise FileNotFoundError(f"Folder {queries_folder} does not exist")
    # Sort the path according to the utm coordinates
    database_paths = glob(join(database_folder, "**", "*.jpg"), recursive=True)
    database_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
    queries_paths = glob(join(queries_folder, "**", "*.jpg"),  recursive=True)
    queries_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
    # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
    database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
    queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)
    # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    soft_positives_per_query = knn.radius_neighbors(queries_utms,
                                                    radius=val_positive_dist_threshold,
                                                    return_distance=False)
    for i in range(len(soft_positives_per_query)):
        soft_positives_per_query[i].sort()
    # Save the dataset in cache/datasets/{dataset_name}{suffix}/
    if not os.path.exists(f"cache/datasets/{dataset_name}{suffix}/"):
        os.makedirs(f"cache/datasets/{dataset_name}{suffix}/")
    # Save the dataset in .npy format
    np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_dbImages.npy"), database_paths)
    np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_qImages.npy"), queries_paths)
    np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_gt.npy"), soft_positives_per_query, allow_pickle=True)

def save_dataset_npy_sfxl(database_folder, queries_folder, dataset_name, split, val_positive_dist_threshold, suffix=""):
    if not os.path.exists(database_folder):
        raise FileNotFoundError(f"Folder {database_folder} does not exist")
    if not os.path.exists(queries_folder):
        raise FileNotFoundError(f"Folder {queries_folder} does not exist")
    # Sort the path according to the utm coordinates
    if split == "train":
        database_paths = glob(join(database_folder, "**", "*.jpg"), recursive=True)
        database_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        soft_positives_per_query = knn.radius_neighbors(database_utms,
                                                        radius=val_positive_dist_threshold,
                                                        return_distance=False)
        for i in range(len(soft_positives_per_query)):
            soft_positives_per_query[i].sort()
        # Save the dataset in cache/datasets/{dataset_name}{suffix}/
        if not os.path.exists(f"cache/datasets/{dataset_name}{suffix}/"):
            os.makedirs(f"cache/datasets/{dataset_name}{suffix}/")
        # Save the dataset in .npy format
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_dbImages.npy"), database_paths)
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_gt.npy"), soft_positives_per_query, allow_pickle=True)
    elif split == "val": # Single level
        database_paths = glob(join(database_folder, "*.jpg"), recursive=True)
        database_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
        queries_paths = glob(join(queries_folder, "*.jpg"),  recursive=True)
        queries_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
        queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        soft_positives_per_query = knn.radius_neighbors(queries_utms,
                                                        radius=val_positive_dist_threshold,
                                                        return_distance=False)
        for i in range(len(soft_positives_per_query)):
            soft_positives_per_query[i].sort()
        # Save the dataset in cache/datasets/{dataset_name}{suffix}/
        if not os.path.exists(f"cache/datasets/{dataset_name}{suffix}/"):
            os.makedirs(f"cache/datasets/{dataset_name}{suffix}/")
        # Save the dataset in .npy format
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_dbImages.npy"), database_paths)
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_qImages.npy"), queries_paths)
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_gt.npy"), soft_positives_per_query, allow_pickle=True)
    elif split == "test": # Single level
        database_paths = glob(join(database_folder, "**", "*.jpg"), recursive=True)
        database_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
        queries_paths = glob(join(queries_folder, "*.jpg"),  recursive=True)
        queries_paths.sort(key=lambda x: (float(x.split("@")[1]), float(x.split("@")[2])))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
        queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        soft_positives_per_query = knn.radius_neighbors(queries_utms,
                                                        radius=val_positive_dist_threshold,
                                                        return_distance=False)
        for i in range(len(soft_positives_per_query)):
            soft_positives_per_query[i].sort()
        # Save the dataset in cache/datasets/{dataset_name}{suffix}/
        if not os.path.exists(f"cache/datasets/{dataset_name}{suffix}/"):
            os.makedirs(f"cache/datasets/{dataset_name}{suffix}/")
        # Save the dataset in .npy format
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_dbImages.npy"), database_paths)
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_qImages.npy"), queries_paths)
        np.save(join(f"cache/datasets/{dataset_name}{suffix}/", f"{dataset_name}_{split}_gt.npy"), soft_positives_per_query, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset in .npy format")
    parser.add_argument("--datasets_folder", type=str, default="datasets", help="Folder containing the datasets")
    parser.add_argument("--dataset_name", type=str, default="Nordland", help="Name of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    args = parser.parse_args()
    generate_dataset_npy(args.datasets_folder, args.dataset_name, args.split)
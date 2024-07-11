import os
import numpy as np
from glob import glob
from os.path import join
from sklearn.neighbors import NearestNeighbors
import shutil
import pandas as pd
from collections import defaultdict

def load_npy_to_df(dataset_name):
    db = np.load(os.path.join(NPY_ROOT, dataset_name, f"{dataset_name}_train_dbImages.npy"))
    db = pd.DataFrame(db, columns=["key"])
    db.insert(0, 'query', False)
    easting = db["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = db["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    db.insert(0, 'easting', easting)
    db.insert(0, 'northing', northing)
    q = np.load(os.path.join(NPY_ROOT, dataset_name, f"{dataset_name}_train_qImages.npy"))
    q = pd.DataFrame(q, columns=["key"])
    q.insert(0, 'query', True)
    easting = q["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = q["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    q.insert(0, 'easting', easting)
    q.insert(0, 'northing', northing)
    df = pd.concat([db, q], ignore_index=True)
    df.insert(0, 'unique_cluster', -1)
    return df

# From Cosplace
def initialize(dataset_folder, M=10, N=5, alpha=30, L=2, min_images_per_class=10, filename=None):
    print(f"Searching training images in {dataset_folder}")
    
    images_paths = read_images_paths(dataset_folder)
    print(f"Found {len(images_paths)} images")
    
    print("For each image, get its UTM east, UTM north and heading from its path")
    images_metadatas = [p.split("@") for p in images_paths]
    # field 1 is UTM east, field 2 is UTM north, field 9 is heading
    utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
    utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)
    
    print("For each image, get class and group to which it belongs")
    class_id__group_id = [get__class_id__group_id(*m, M, alpha, N, L)
                            for m in utmeast_utmnorth_heading]
    
    print("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)
    
    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}
    
    print("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)
    
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]
    
    return (classes_per_group, images_per_class)

def read_images_paths(dataset_folder, get_abs_path=False):
    """Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [os.path.join(dataset_folder, path) for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        print(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")
    
    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p[len(dataset_folder) + 1:] for p in images_paths]
    
    return images_paths

def get__class_id__group_id(utm_east, utm_north, heading, M, alpha, N, L):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north and
            heading (e.g. (396520, 4983800,120)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        rounded_heading = int(heading // alpha * alpha)
        
        class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)
        # group_id goes from (0, 0, 0) to (N, N, L)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M,
                    rounded_heading % (alpha * L) // alpha)
        return class_id, group_id
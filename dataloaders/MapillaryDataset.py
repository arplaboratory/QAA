from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.

class MapillaryDataset(Dataset):
    def __init__(self, split, input_transform = None):
        
        self.dataset_name = "mapillary_sls"
        self.input_transform = input_transform
        self.split = split
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_dbImages.npy')
        
        # hard coded query image names.
        self.qImages = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_qImages.npy')
        
        # hard coded index of query images
        self.qIdx = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_qIdx.npy')
        
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_pIdx.npy', allow_pickle=True)
        
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
class MapillaryTestDataset(Dataset):
    def __init__(self, split, input_transform = None):
        
        self.dataset_name = "mapillary_sls"
        self.input_transform = input_transform
        self.split = split
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_dbImages.npy')
        
        # hard coded query image names.
        self.qImages = np.load(f'./cache/datasets/mapillary_sls/msls_{split}_qImages.npy')
        
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        # ground truth
        self.ground_truth = None

        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                f.write(f"{q} {db}\n")
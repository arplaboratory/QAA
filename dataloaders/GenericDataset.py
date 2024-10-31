from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision

# NOTE: Only for validation purposes
NPY_ROOT = 'cache/datasets/'

class GenericDataset(Dataset):
    def __init__(self, dataset_name, split, input_transform = None, backup_transform = None):
        
        self.dataset_name = dataset_name
        self.input_transform = input_transform
        self.backup_transform = backup_transform
        self.split = split

        # reference images names
        self.dbImages = np.load(NPY_ROOT + f'{dataset_name}/{dataset_name}_{split}_dbImages.npy')
        
        # query images names
        self.qImages = np.load(NPY_ROOT + f'{dataset_name}/{dataset_name}_{split}_qImages.npy')
        
        # ground truth
        self.ground_truth = np.load(NPY_ROOT + f'{dataset_name}/{dataset_name}_{split}_gt.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    
    def __getitem__(self, index):
        if self.dataset_name == "nordland_subset":
            img = torchvision.io.read_image("datasets/nordland_subset/" + self.images[index]) # Add prefix
        else:
            img = torchvision.io.read_image(self.images[index])

        if self.input_transform:
            try:
                img = self.input_transform(img)
            except Exception as e:
                # Grayscale images
                img = self.backup_transform(img)

        return img, index, self.images[index]

    def __len__(self):
        return len(self.images)
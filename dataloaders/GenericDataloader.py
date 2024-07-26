import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
import torch
import os

from prettytable import PrettyTable

from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.GenericDataset import GenericDataset
from dataloaders.MapillaryDataset import MapillaryDataset, MapillaryTestDataset
from dataloaders.CliqueMapillaryDataset import CliqueMapillaryDataset
from dataloaders.CliqueGenericDataset import CliqueGenericDataset
from dataloaders.CliqueSFXLDataset import CliqueSFXLDataset
from utils.load_cfg import load_datasets_config
import wandb

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_batch_size=30,
                 test_batch_size=256,
                 train_image_size=(480, 640),
                 test_image_size=(480, 640),
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 dataset_names=None,
                 train_cfg_training=None
                 ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_image_size = train_image_size
        self.test_image_size = test_image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.train_dataset_names = dataset_names.train_datasets
        self.val_dataset_names = dataset_names.val_datasets
        self.test_dataset_names = dataset_names.test_datasets
        self.train_datasets_cfg = load_datasets_config(self.train_dataset_names)
        self.val_datasets_cfg = load_datasets_config(self.val_dataset_names)
        self.test_datasets_cfg = load_datasets_config(self.test_dataset_names)
        self.train_cfg_training = train_cfg_training
        if self.train_cfg_training.recompute_clusters and self.train_cfg_training.recompute_interval!=0:
            self.recompute_count = 0
        self.model = None

        self.train_transform = T.Compose([
            T.Resize(self.train_image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(self.test_image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])
        
        self.test_transform = T.Compose([
            T.Resize(self.test_image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])
        
        self.test_grayscale_transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(self.test_image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])
        
        self.train_loader_config_general = {
            'batch_size': self.train_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False} # Shuffle is done in CliqueMining reload
    
        if "GSV" in self.train_dataset_names:
            self.train_loader_config_GSV = {
                'batch_size': self.train_batch_size,
                'num_workers': self.num_workers,
                'drop_last': False,
                'pin_memory': True,
                'shuffle': self.train_datasets_cfg["GSV"].training.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

        self.test_loader_config = {
            'batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}


        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader (pitts_train, msls_train, ...etc)
            self.train_datasets = []
            for dataset_name in self.train_dataset_names:
                assert self.train_datasets_cfg[dataset_name].training.available
                if dataset_name == "GSV":
                    GSV_params = self.train_datasets_cfg[dataset_name]
                    self.train_datasets.append(GSVCitiesDataset(
                                                split="train",
                                                cities=GSV_params.training.GSV_TRAIN_CITIES,
                                                img_per_place=GSV_params.training.img_per_place,
                                                min_img_per_place=GSV_params.training.min_img_per_place,
                                                random_sample_from_each_place=GSV_params.training.random_sample_from_each_place,
                                                transform=self.train_transform))
                    if GSV_params.training.show_data_stats:
                        self.print_GSV_stats(self.train_datasets[-1], GSV_params)
                elif dataset_name == "mapillary_sls":
                    self.train_datasets.append(CliqueMapillaryDataset(
                                                split="train", 
                                                transform=self.train_transform,
                                                batch_size=self.train_batch_size,
                                                only_top_k=self.train_cfg_training.only_top_k,
                                                shuffle_method=self.train_cfg_training.shuffle_method,
                                                prefetch_factor=self.train_cfg_training.prefetch_factor,
                                                **self.train_datasets_cfg["mapillary_sls"].training.clique_args))
                elif dataset_name == "SF_XL":
                    self.train_datasets.append(CliqueSFXLDataset(
                                                split="train",
                                                transform=self.train_transform,
                                                batch_size=self.train_batch_size,
                                                only_top_k=self.train_cfg_training.only_top_k,
                                                shuffle_method=self.train_cfg_training.shuffle_method,
                                                prefetch_factor=self.train_cfg_training.prefetch_factor,
                                                **self.train_datasets_cfg["SF_XL"].training.clique_args))
                else:
                    self.train_datasets.append(CliqueGenericDataset(
                                                dataset_name=dataset_name, 
                                                split="train", 
                                                transform=self.train_transform,
                                                batch_size=self.train_batch_size,
                                                only_top_k=self.train_cfg_training.only_top_k,
                                                **self.train_datasets_cfg[dataset_name].training.clique_args))
                print(f'Dataset {dataset_name} loaded: Length = {len(self.train_datasets[-1])}')

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                assert self.val_datasets_cfg[dataset_name].validation.available
                if dataset_name == "mapillary_sls":
                    self.val_datasets.append(MapillaryDataset(split="val", input_transform=self.valid_transform))
                else:
                    self.val_datasets.append(GenericDataset(dataset_name=dataset_name, split="val", input_transform=self.valid_transform))
            wandb.config.update({'train_datasets': self.train_datasets_cfg, 'val_datasets': self.val_datasets_cfg, 'test_datasets': self.test_datasets_cfg})

        elif stage=="test":
            # load test sets (pitts_val, msls_val, ...etc)
            self.test_datasets = []
            for dataset_name in self.test_dataset_names:
                assert self.test_datasets_cfg[dataset_name].test.available
                if dataset_name == "mapillary_sls":
                    self.test_datasets.append(MapillaryTestDataset(split="test", input_transform=self.test_transform))
                else:
                    self.test_datasets.append(GenericDataset(dataset_name=dataset_name, split="test", input_transform=self.test_transform, backup_transform=self.test_grayscale_transform))

        elif stage=="validate":
            # load test sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                assert self.val_datasets_cfg[dataset_name].validation.available
                if dataset_name == "mapillary_sls":
                    self.val_datasets.append(MapillaryDataset(split="val", input_transform=self.valid_transform))
                else:
                    self.val_datasets.append(GenericDataset(dataset_name=dataset_name, split="val", input_transform=self.valid_transform))

    def reload(self, dataset_name, index):
        if dataset_name == "GSV":
            GSV_params = self.train_datasets_cfg["GSV"]
            self.train_datasets[index] = GSVCitiesDataset(split="train",
                                                        cities=GSV_params.training.GSV_TRAIN_CITIES,
                                                        img_per_place=GSV_params.training.img_per_place,
                                                        min_img_per_place=GSV_params.training.min_img_per_place,
                                                        random_sample_from_each_place=GSV_params.training.random_sample_from_each_place,
                                                        transform=self.train_transform)
        else:
            if self.train_cfg_training.recompute_clusters and self.train_cfg_training.recompute_interval!=0 and self.recompute_count == self.train_cfg_training.recompute_interval:
                print("RECOMPUTE")
                self.train_datasets[index].reload(model=self.model, recompute=True)
                self.recompute_count = 0
            else:
                print("SHUFFLE")
                self.train_datasets[index].reload(model=self.model, recompute=False)
                if self.train_cfg_training.recompute_clusters and self.train_cfg_training.recompute_interval!=0:
                    self.recompute_count += 1

    def train_dataloader(self):
        train_dataloaders = {}
        for index, train_dataset_name in enumerate(self.train_dataset_names):
            print("Reloading to shuffle")
            self.reload(train_dataset_name, index) # Following reload routine to shuffle cities
            train_dataset = self.train_datasets[index]
            if train_dataset_name == "GSV":
                train_dataloaders[train_dataset_name] = DataLoader(
                    dataset=train_dataset, **self.train_loader_config_GSV)
            else:
                train_dataloaders[train_dataset_name] = DataLoader(
                    dataset=train_dataset, **self.train_loader_config_general)
        print(f"Train dataloaders: {train_dataloaders}")
        return train_dataloaders

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders
    
    def test_dataloader(self):
        test_dataloaders = []
        for test_dataset in self.test_datasets:
            test_dataloaders.append(DataLoader(
                dataset=test_dataset, **self.test_loader_config))
        return test_dataloaders

    def print_GSV_stats(self, GSV_dataset, GSV_params):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(GSV_params.training.GSV_TRAIN_CITIES)}"])
        table.add_row(["# of places", f'{GSV_dataset.__len__()}'])
        table.add_row(["# of images", f'{GSV_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.train_batch_size}x{GSV_params.training.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{GSV_dataset.__len__()//self.train_batch_size}"])
        table.add_row(["Image size", f"{self.train_image_size}"])
        print(table.get_string(title="Training config"))
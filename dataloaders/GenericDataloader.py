import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from prettytable import PrettyTable

from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.GenericDataset import GenericDataset
from dataloaders.MapillaryDataset import MapillaryDataset
from dataloaders.CliqueMiningDataset import CliqueMapillaryDataset
from utils.load_cfg import load_datasets_config

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 image_size=(480, 640),
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 dataset_names=None
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.train_dataset_names = dataset_names.train_datasets
        self.val_dataset_names = dataset_names.val_datasets
        self.train_datasets_cfg = load_datasets_config(self.train_dataset_names)
        self.val_datasets_cfg = load_datasets_config(self.val_dataset_names)

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config_general = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False} # Shuffle is done in CliqueMining reload
    
        if "GSV" in self.train_dataset_names:
            self.train_loader_config_GSV = {
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'drop_last': False,
                'pin_memory': True,
                'shuffle': self.train_datasets_cfg["GSV"].training.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader (pitts_train, msls_train, ...etc)
            self.train_datasets = []
            for dataset_name in self.train_dataset_names:
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
                                                **self.train_datasets_cfg["mapillary_sls"].training.clique_mapillary_args))
                else:
                    self.train_datasets.append(GenericDataset(dataset_name=dataset_name, split="train", input_transform=self.train_transform))

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for dataset_name in self.val_dataset_names:
                if dataset_name == "mapillary_sls":
                    self.val_datasets.append(MapillaryDataset(split="val", input_transform=self.valid_transform))
                else:
                    self.val_datasets.append(GenericDataset(dataset_name=dataset_name, split="val", input_transform=self.valid_transform))


    def reload(self, dataset_name):
        if dataset_name == "GSV":
            GSV_params = self.train_datasets_cfg["GSV"]
            self.train_datasets[self.current_train_dataset_index] = GSVCitiesDataset(
                                                                    split="train",
                                                                    cities=GSV_params.training.GSV_TRAIN_CITIES,
                                                                    img_per_place=GSV_params.training.img_per_place,
                                                                    min_img_per_place=GSV_params.training.min_img_per_place,
                                                                    random_sample_from_each_place=GSV_params.training.random_sample_from_each_place,
                                                                    transform=self.train_transform)
        else:
            self.train_datasets[self.current_train_dataset_index].reload()

    def train_dataloader(self):
        if not hasattr(self, "current_train_dataset_index"):
            self.current_train_dataset_index = 0
        else:
            self.current_train_dataset_index += 1
            if self.current_train_dataset_index >= len(self.train_datasets):
                self.current_train_dataset_index = 0
        train_dataset_name = self.train_dataset_names[self.current_train_dataset_index]
        self.reload(train_dataset_name) # Following reload routine to shuffle cities
        train_dataset = self.train_datasets[self.current_train_dataset_index]
        if train_dataset_name == "GSV":
            train_dataloaders = DataLoader(
                dataset=train_dataset, **self.train_loader_config_GSV)
        else:
            train_dataloaders = DataLoader(
                dataset=train_dataset, **self.train_loader_config_general)
        return train_dataloaders

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

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

        # table = PrettyTable()
        # table.field_names = ['Data', 'Value']
        # table.align['Data'] = "l"
        # table.align['Value'] = "l"
        # table.header = False
        # for i, val_set_name in enumerate(self.val_set_names):
        #     table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        # # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        # print(table.get_string(title="Validation Datasets"))
        # print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{GSV_params.training.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{GSV_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
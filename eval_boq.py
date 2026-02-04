import pytorch_lightning as pl
import argparse

from utils.load_cfg import load_config
from dataloaders.GenericDataloader import GenericDataModule
from vpr_model import VPRModel
import utils
import torch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models

class GenericModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.faiss_gpu = False
    def forward(self, x):
        return self.model(x)
    
    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors, attn = self(places)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.test_calculate_recall(self.current_dataloader_idx)
            self.current_dataloader_idx = dataloader_idx
        self.test_outputs.append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_test_epoch_start(self):
        # reset the outputs list
        self.test_outputs = []
        self.results_list = []
        self.current_dataloader_idx = 0
    
    def on_test_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        self.test_calculate_recall(self.current_dataloader_idx) # For last dataset
        for i, test_dataset in enumerate(dm.test_datasets):
            test_set_name = test_dataset.dataset_name
            pitts_dict = self.results_list[i]
            if pitts_dict == []:
                continue
            self.log(f'{test_set_name}_{test_dataset.split}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{test_set_name}_{test_dataset.split}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{test_set_name}_{test_dataset.split}/R10', pitts_dict[10], prog_bar=False, logger=True)        
        print('\n\n')
        # reset the outputs list
        self.test_outputs = []
        self.results_list = []

    def test_calculate_recall(self, dataloader_idx):
        # Clean memory once one dataset finished
        test_step_outputs = self.test_outputs
        dm = self.trainer.datamodule
        test_dataset = dm.test_datasets[dataloader_idx]
        test_set_name = test_dataset.dataset_name
        feats = torch.concat(test_step_outputs, dim=0)
            
        if test_set_name == "mapillary_sls":
            # split to ref and queries
            num_references = test_dataset.num_references
            positives = None
            testing = True # This flag is for msls

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            preds = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=test_set_name,
                faiss_gpu=self.faiss_gpu,
                testing=testing,
            )
            del r_list, q_list, feats, num_references, positives
            assert test_dataset.split == "test"
            print(f"Save predictions to msls_preds.txt")
            try:
                test_dataset.save_predictions(preds, f'QAA/{self.logger.version}/checkpoints/msls_preds.txt')
            except Exception:
                print("MSLS PRED TEXT SAVE IN ROOT FOLDER")
                test_dataset.save_predictions(preds, f'./msls_preds.txt')
            self.results_list.append([])
        else:
            num_references = test_dataset.num_references
            positives = test_dataset.ground_truth
            testing = False # This flag is for other dataset that has ground truth

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=test_set_name,
                faiss_gpu=self.faiss_gpu,
                testing=testing,
            )
            del r_list, q_list, feats, num_references, positives
            self.results_list.append(pitts_dict)

        self.test_outputs = []

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    # wandb_logger = WandbLogger(name=args.config.split('/')[-1].split('.')[0], project="QAA")
    datamodule = GenericDataModule(
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        test_image_size=train_cfg.training.test_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
    )
    
    # Change this to load other models
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288)
    model = GenericModel(model)

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        max_epochs=train_cfg.training.num_epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.test(model=model, datamodule=datamodule)
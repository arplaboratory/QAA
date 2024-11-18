import pytorch_lightning as pl
import argparse
import os

from vpr_model import VPRModel
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args.add_argument('--ckpt_path', type=str)
    args.add_argument('--visualize', action='store_true')
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    # wandb_logger = WandbLogger(name=args.config.split('/')[-1].split('.')[0], project="UniVPR")
    datamodule = GenericDataModule(
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        test_image_size=train_cfg.training.test_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
    )
    
    model = VPRModel.load_from_checkpoint(args.ckpt_path)
    if args.visualize:
        model.visualize = True
        if os.path.isdir('vis'):
            raise ValueError('Visualisation directory does exist')
        else:
            os.mkdir('vis')

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
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
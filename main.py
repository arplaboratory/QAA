import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

from vpr_model import VPRModel
from utils.load_cfg import load_config, load_datasets_config
from dataloaders.GenericDataloader import GenericDataModule

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # For downloading the pretrained models

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str)
    args = args.parse_args()
    # we load the training configuration
    train_cfg = load_config(args.config)
    wandb_logger = WandbLogger(name=args.config.split('/')[-1].split('.')[0], project="UniVG")
    datamodule = GenericDataModule(
        train_batch_size=train_cfg.training.train_batch_size,
        test_batch_size=train_cfg.training.test_batch_size,
        train_image_size=train_cfg.training.train_image_size,
        test_image_size=train_cfg.training.test_image_size,
        num_workers=train_cfg.training.num_workers,
        dataset_names=train_cfg.datasets,
        train_cfg_training=train_cfg.training,
    )
    
    model = VPRModel(
        #---- Encoder
        backbone_arch=train_cfg.model.backbone_arch,
        backbone_config=train_cfg.model.backbone_config,
        agg_arch=train_cfg.model.agg_arch,
        agg_config=train_cfg.model.agg_config,
        lr=train_cfg.training.optimizer["lr"],
        optimizer=train_cfg.training.optimizer["name"],
        weight_decay=train_cfg.training.optimizer["weight_decay"], # 0.001 for sgd and 0 for adam,
        momentum=train_cfg.training.optimizer["momentum"],
        lr_sched=train_cfg.training.scheduler["name"],
        lr_sched_args = train_cfg.training.scheduler["args"],

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=train_cfg.training.loss["name"],
        miner_name=train_cfg.training.miner["name"], # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=train_cfg.training.miner["margin"],
        faiss_gpu=train_cfg.training.faiss_gpu
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=f'{train_cfg.datasets.target_val_dataset}_val/R1',
        filename=f'{model.encoder_arch}' + '_{epoch:02d}_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    #------------------
    # we instanciate a trainer
    pl.seed_everything(42, worker=True)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=train_cfg.training.num_epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb, lr_monitor],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        logger=wandb_logger,
        detministic=True,
    )

    # we call the trainer, we give it the model and the datamodule
    # trainer.validate(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
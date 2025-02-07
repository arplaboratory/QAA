import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import torchvision
import torch.nn.functional as F
import os

import utils
from utils.measure_flop import measure_flop
from models import helper
from dataloaders.GenericDataloader import IMAGENET_MEAN_STD
from PIL import Image
from matplotlib import pyplot as plt

imagenet_mean = IMAGENET_MEAN_STD['mean']
imagenet_std = IMAGENET_MEAN_STD['std']
inv_base_transform = torchvision.transforms.Compose(
    [ 
        torchvision.transforms.Normalize(mean = [ -m/s for m, s in zip(imagenet_mean, imagenet_std)],
                             std = [ 1/s for s in imagenet_std]),
    ]
)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        backbone_lr=0.03,
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False,
        cross_loss=False,
        cross_loss_weight=1.0,
        recompute_desc=False,
        decorrelation="none",
        decorrelation_loss_weight=1.0,
        decorrelation_off_lambda=0.005,
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = float(lr)
        self.backbone_lr = float(backbone_lr)
        self.optimizer = optimizer
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        self.cross_loss = cross_loss
        self.cross_loss_weight = cross_loss_weight
        self.recompute_desc = recompute_desc
        self.decorrelation = decorrelation
        self.decorrelation_loss_weight = decorrelation_loss_weight
        self.decorrelation_off_lambda = decorrelation_off_lambda
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        if agg_arch == "QueriesSALAD":
            agg_arch = "QAA"
        if agg_arch == "ConditionedQueriesSALAD":
            agg_arch = "ConditionedQAA"
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        self.log_img_first_iter = False
        self.visualize = False
        measure_flop(self)
        
    # the forward pass of the lightning model
    def forward(self, x, domain_idx=None):
        x = self.backbone(x, domain_idx=domain_idx)
        if self.visualize and "Queries" not in self.agg_arch:
            raise NotImplementedError()
        if self.decorrelation == "none" or domain_idx == None:
            x = self.aggregator(x, domain_idx=domain_idx, visualize=self.visualize)
        else:
            x = self.aggregator(x, domain_idx=domain_idx, visualize=self.visualize, decorrelation=self.decorrelation)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if not hasattr(self.aggregator, "freeze") or self.aggregator.freeze == "none":
            params = [{'params': self.aggregator.parameters()}]
            print(f"Add params: aggregator")
        else:
            params = [{'params': self.aggregator.token_features.parameters()}]
            print(f"Add params: aggregator - token_features")
            params = params + [{'params': self.aggregator.score.parameters()}]
            print(f"Add params: aggregator - score")
            if hasattr(self.aggregator, "cluster_feature"):
                params = params + [{'params': self.aggregator.cluster_feature.parameters()}]
                print(f"Add params: aggregator - feature_cross_attn")
            if self.aggregator.freeze == "feature":
                params = params + [{'params': self.aggregator.queries_score.parameters()}]
                print(f"Add params: aggregator - queries_score")
            elif self.aggregator.freeze == "score":
                params = params + [{'params': self.aggregator.queries_feature.parameters()}]
                print(f"Add params: aggregator - queries_feature")
            else:
                raise NotImplementedError()
        if hasattr(self.backbone, "domain_prompt_model"):
            params.append({'params': self.backbone.domain_prompt_model.parameters()})
            print(f"Add params: domain_prompt_model")
        if hasattr(self.backbone, "domain_prompt_model_list"):
            params.append({'params': self.backbone.domain_prompt_model_list.parameters()})
            print(f"Add params: domain_prompt_model_list")
        if hasattr(self.backbone, "domain_prompt_mlp_list"):
            params.append({'params': self.backbone.domain_prompt_mlp_list.parameters()})
            print(f"Add params: domain_prompt_mlp_list")
        if hasattr(self.backbone, "domain_prompt_mlp"):
            params.append({'params': self.backbone.domain_prompt_mlp.parameters()})
            print(f"Add params: domain_prompt_mlp")
        if hasattr(self.backbone, "shared_prompt_mlp"):
            params.append({'params': self.backbone.shared_prompt_mlp.parameters()})
            print(f"Add params: shared_prompt_mlp")
        if self.backbone_config['num_trainable_blocks'] > 0:
            for i, blk in enumerate(self.backbone.model.blocks[-self.backbone_config['num_trainable_blocks']:]):
                params.append({'params': blk.parameters(), 'lr': self.backbone_lr})
                print (f"Add params: Trainable block {len(self.backbone.model.blocks) - self.backbone_config['num_trainable_blocks'] + i}")
        else:
            print("All blocks are frozen")
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Define how a single optimization step is executed.

        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            optimizer: Optimizer instance.
            optimizer_closure: Closure for the optimizer.
        """
        if self.trainer.global_step < self.lr_sched_args['total_iters']:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_sched_args['total_iters'])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg["initial_lr"]

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        prev_label = -1
        images = []
        labels = []
        domain_idx = []
        BS_list = []
        N_list = []
        for i, train_dataset_name in enumerate(batch.keys()):
            places_single = batch[train_dataset_name][0]
            labels_single = batch[train_dataset_name][1]
            domain_idx_single = torch.ones(places_single.shape[0] * places_single.shape[1], dtype=torch.long) * i
            domain_idx_single = domain_idx_single.to(places_single.device).long()
            if not self.log_img_first_iter:
                mean_tensor = torch.Tensor(IMAGENET_MEAN_STD['mean']).view(1, 1, 3, 1, 1)
                std_tensor = torch.Tensor(IMAGENET_MEAN_STD['std']).view(1, 1, 3, 1, 1)
                denormalized_image = places_single.cpu() * std_tensor + mean_tensor
                denormalized_image = torch.clamp(denormalized_image, 0, 1)
                list_images = [img for img in denormalized_image.view(-1, denormalized_image.shape[-3], denormalized_image.shape[-2], denormalized_image.shape[-1])]
                list_images = list_images[:16]
                self.logger.log_image(f'input_images_{train_dataset_name}', list_images)
            labels_single += prev_label + 1
            prev_label = labels_single.max()
            BS, N, ch, h, w = places_single.shape
            BS_list.append(BS)
            N_list.append(N)
            images.append(places_single.view(-1, ch, h, w))
            labels.append(labels_single.view(-1))
            domain_idx.append(domain_idx_single.view(-1))
        self.log_img_first_iter = True

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        domain_idx = torch.cat(domain_idx, dim=0)

        # Feed forward the batch to the model
        if self.backbone.domain_prompt != "none":
            descriptors, domain_desc = self(images, domain_idx)
        elif self.decorrelation != "none":
            descriptors, query_p, query_f = self(images, domain_idx)
        else:
            descriptors = self(images, domain_idx) # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = 0
        domain_loss = 0
        for i in range(len(BS_list)):
            BS = BS_list[i]
            N = N_list[i]
            if i == 0:
                desc = descriptors[:BS*N]
                lab = labels[:BS*N]
            else:
                desc = descriptors[prev_BS_N:prev_BS_N+BS*N]
                lab = labels[prev_BS_N:prev_BS_N+BS*N]
            loss += self.loss_function(desc, lab)
            if self.cross_loss:
                if self.backbone.multi_adapt == "none":
                    if i == 0:
                        desc = domain_desc[:BS*N]
                        lab = labels[:BS*N]
                    else:
                        desc = domain_desc[prev_BS_N:prev_BS_N+BS*N]
                        lab = labels[prev_BS_N:prev_BS_N+BS*N]
                    domain_loss = self.loss_function(desc, lab)
                else:
                    domain_loss = 0
                    for j in range(len(domain_desc)):
                        if i == 0:
                            desc = domain_desc[j][:BS*N]
                            lab = labels[:BS*N]
                        else:
                            desc = domain_desc[j][prev_BS_N:prev_BS_N+BS*N]
                            lab = labels[prev_BS_N:prev_BS_N+BS*N]
                        domain_loss += self.loss_function(desc, lab)
                loss += self.cross_loss_weight * domain_loss
            if i == 0:
                prev_BS_N = BS*N
            else:
                prev_BS_N += BS*N
        
        if self.decorrelation != "none":
            if self.decorrelation == "score":
                loss += self.decorrelation_loss_weight * self.barlow_loss_rows(query_p, lambda_offdiag=self.decorrelation_off_lambda)
            elif self.decorrelation == "feature":
                loss += self.decorrelation_loss_weight * self.barlow_loss_rows(query_f, lambda_offdiag=self.decorrelation_off_lambda)
            elif self.decorrelation == "both":
                loss += self.decorrelation_loss_weight * self.barlow_loss_rows(query_p, lambda_offdiag=self.decorrelation_off_lambda)
                loss += self.decorrelation_loss_weight * self.barlow_loss_rows(query_f, lambda_offdiag=self.decorrelation_off_lambda)
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        if self.cross_loss:
            self.log('domain_loss', domain_loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []
        # self.log_img_first_iter = False

    def _visualize_batch(self, dataset_name, batch_idx, places, score):
        if not os.path.isdir(f'vis/{dataset_name}'):
            os.mkdir(f'vis/{dataset_name}')
        os.mkdir(f'vis/{dataset_name}/{batch_idx}')
        attn_map = (attn_map - attn_map.min())/(attn_map.max() - attn_map.min())
        attn_map = attn_map.view(-1, self.aggregator.num_queries, places.shape[-2]//14, places.shape[-1]//14)
        attn_map = F.interpolate(attn_map, size=(places.shape[-2], places.shape[-1]), mode='bilinear', align_corners=True)
        for i in range(len(places)):
            ndarr = inv_base_transform(places[i]).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(f'vis/{dataset_name}/{batch_idx}/place_{i}.png')
            for j in range(len(attn_map[i])):
                ndarr_attn = attn_map[i][j].unsqueeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                plt.figure()
                plt.imshow(ndarr)
                plt.imshow(ndarr_attn, alpha=0.5, cmap='jet')
                plt.axis('off')
                plt.savefig(f'vis/{dataset_name}/{batch_idx}/attn_map_{i}_{j}.png', bbox_inches='tight', transparent="True", pad_inches=0)
                plt.close()
            # Calculate the similarity matrix between scores of current place and current place
            plt.figure()
            plt.imshow(torch.matmul(score[i][0].T, score[i][0]).cpu().numpy(), cmap='viridis')
            plt.axis('off')
            plt.savefig(f'vis/{dataset_name}/{batch_idx}/score_sim_matrix_{i}.png', bbox_inches='tight', transparent="True", pad_inches=0)
            plt.close()
            # Calculate the similarity matrix between scores of 0-th place and current place
            plt.figure()
            plt.imshow(torch.matmul(score[i][0].T, score[0][0]).cpu().numpy(), cmap='viridis')
            plt.axis('off')
            plt.savefig(f'vis/{dataset_name}/{batch_idx}/score_0_sim_matrix_{i}.png', bbox_inches='tight', transparent="True", pad_inches=0)
            plt.close()
    
    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        if self.backbone.domain_prompt != "none":
            descriptors, domain_desc = self(places)
        elif self.visualize:
            descriptors, attn_map, score = self(places)
            dataset_name = self.trainer.datamodule.val_datasets[dataloader_idx].dataset_name
            self._visualize_batch(dataset_name, batch_idx, places, score)
        else:
            descriptors = self(places)
        if dataloader_idx is None: # Only one val dataset
            dataloader_idx = 0
        if self.current_dataloader_idx != dataloader_idx:
            self.val_calculate_recall(self.current_dataloader_idx)
            self.current_dataloader_idx = dataloader_idx
        self.val_outputs.append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = []
        self.results_list = []
        self.current_dataloader_idx = 0
        if self.visualize:            
            self.visualize_queries()
        if self.agg_arch == "QAA":
            self.aggregator.cache_query()
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        self.val_calculate_recall(self.current_dataloader_idx) # For last dataset
        for i, val_dataset in enumerate(dm.val_datasets):
            val_set_name = val_dataset.dataset_name
            pitts_dict = self.results_list[i]
            self.log(f'{val_set_name}_{val_dataset.split}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}_{val_dataset.split}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}_{val_dataset.split}/R10', pitts_dict[10], prog_bar=False, logger=True)        
        print('\n\n')
        # reset the outputs list
        self.val_outputs = []
        self.results_list = []

        if self.recompute_desc:
            print("Update model for recomputing if recomputing is enabled")
            self.trainer.datamodule.model = self # Not sure if this is correct
        else:
            print("Use existing descriptors for recomputing if recomputing is enabled")
            self.trainer.datamodule.model = None

        if self.agg_arch == "QAA":
            self.aggregator.clean_cache()

    def val_calculate_recall(self, dataloader_idx):
        # Clean memory once one dataset finished
        val_step_outputs = self.val_outputs
        dm = self.trainer.datamodule
        val_dataset = dm.val_datasets[dataloader_idx]
        val_set_name = val_dataset.dataset_name
        feats = torch.concat(val_step_outputs, dim=0)
        
        if val_set_name == "mapillary_sls":
            # split to ref and queries
            num_references = val_dataset.num_references
            positives = val_dataset.pIdx
        else:
            num_references = val_dataset.num_references
            positives = val_dataset.ground_truth

        r_list = feats[ : num_references]
        q_list = feats[num_references : ]
        pitts_dict = utils.get_validation_recalls(
            r_list=r_list, 
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 50, 100],
            gt=positives,
            print_results=True,
            dataset_name=val_set_name,
            faiss_gpu=self.faiss_gpu
        )
        del r_list, q_list, feats, num_references, positives
        self.results_list.append(pitts_dict)

        self.val_outputs = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        if self.backbone.domain_prompt != "none":
            descriptors, domain_desc = self(places)
        elif self.visualize:
            descriptors, attn_map, score = self(places)
            dataset_name = self.trainer.datamodule.test_datasets[dataloader_idx].dataset_name
            self._visualize_batch(dataset_name, batch_idx, places, score)
        else:
            descriptors = self(places)
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
        if self.visualize:            
            self.visualize_queries()
        if self.agg_arch == "QAA":
            self.aggregator.cache_query()
    
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

        if self.agg_arch == "QAA":
            self.aggregator.clean_cache()

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
                test_dataset.save_predictions(preds, f'UniVPR/{self.logger.version}/checkpoints/msls_preds.txt')
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

    def visualize_queries(self):
        feature_query = self.aggregator.queries_feature()
        score_query = self.aggregator.queries_score()
        max_min_norm = lambda x: (x - x.min())/(x.max() - x.min())
        plt.figure()
        norm_feature_queries = max_min_norm(feature_query[0])
        print(norm_feature_queries.shape)
        plt.imshow(norm_feature_queries.cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f'vis/feature_query.png', bbox_inches='tight', transparent="True", pad_inches=0)
        plt.close()
        plt.figure()
        norm_score_queries = max_min_norm(score_query[0])
        print(norm_score_queries.shape)
        plt.imshow(norm_score_queries.cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f'vis/score_query.png', bbox_inches='tight', transparent="True", pad_inches=0)
        plt.close()
        norm_feature_queries = F.normalize(feature_query[0], p=2, dim=-1)
        sim_matrix = torch.matmul(norm_feature_queries, norm_feature_queries.T)
        plt.figure()
        plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f'vis/sim_matrix_feature.png', bbox_inches='tight', transparent="True", pad_inches=0)
        plt.close()
        norm_score_queries = F.normalize(score_query[0], p=2, dim=-1)
        sim_matrix = torch.matmul(norm_score_queries, norm_score_queries.T)
        plt.figure()
        plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f'vis//sim_matrix_score.png', bbox_inches='tight', transparent="True", pad_inches=0)
        plt.close()

    def barlow_loss_rows(self, score, lambda_offdiag=0.005):
        """
        Barlow-style decorrelation among columns
        of a (L, D) matrix. If L is your 'batch size' and
        D is your 'feature dimension', this penalizes 
        correlation among the D columns across the L samples.
        """
        score = score.squeeze(0)
        L, D = score.shape
        
        # Center (optional) so the mean of each dimension is 0
        score_centered = score - score.mean(dim=0, keepdim=True)
        
        # Covariance/correlation matrix: D x D
        C = (score_centered.T @ score_centered) / L
        
        # On-diagonal: want ~ 1
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        
        # Off-diagonal: want ~ 0
        off_diag = self.off_diagonal(C).pow_(2).sum()
        
        return on_diag + lambda_offdiag * off_diag

    def off_diagonal(self, x):
        """Return flattened view of off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
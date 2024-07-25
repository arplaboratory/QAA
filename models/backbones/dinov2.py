import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from ..aggregators.salad import SALAD

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class ClusterNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight + self.residual_weight
        bias = self.bias + self.residual_bias
        return self.norm(x) * weight.unsqueeze(1) + bias.unsqueeze(1)
    
    def set_residual_weight_bias(self, weight: Tensor, bias: Tensor) -> None:
        self.residual_weight = weight
        self.residual_bias = bias
        
    def set_weight_bias(self, weight: Tensor, bias: Tensor) -> None:
        self.weight = weight
        self.bias = bias

class ClusterLayerScale(nn.Module):
    def __init__(
        self,
        hidden_size,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        gamma = self.gamma + self.residual_gamma
        return x.mul_(gamma.unsqueeze(1)) if self.inplace else x * gamma.unsqueeze(1)

    def set_residual_gamma(self, gamma: Tensor) -> None:
        self.residual_gamma = gamma

    def set_gamma(self, gamma: Tensor) -> None:
        self.gamma = gamma

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False,
            domain_prompt=False,
            num_clusters=64,
            cluster_dim=16,
            token_dim=128,
            dropout=0.0,
            divide=1,
            shared_clusters=0,
            decouple=False,
            padding="detach",
            mlp_nonlinear=False,
            final_layer_norm=True,
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.domain_prompt = domain_prompt
        self.final_layer_norm = final_layer_norm
        if self.domain_prompt:
            hidden_size = self.model.blocks[0].norm1.weight.shape[0]
            assert padding in ["detach", "zero"], 'Padding should be either detach or zero'
            assert self.num_trainable_blocks > 0, 'First blocks should be frozen when using domain prompt'
            self.domain_prompt_model = SALAD(num_channels=hidden_size, num_clusters=num_clusters, cluster_dim=cluster_dim,
                                             token_dim=token_dim, dropout=dropout, padding=padding,
                                             divide=divide, decouple=decouple, shared_clusters=shared_clusters)
            self.domain_prompt_mlp_list = nn.ModuleList()
            for i, blk in enumerate(self.model.blocks[-self.num_trainable_blocks:]):
                if i == self.num_trainable_blocks - 1 and not self.final_layer_norm:
                    self.domain_prompt_mlp_list.append(nn.Sequential(nn.SiLU() if mlp_nonlinear else nn.Identity(),
                                                                nn.Linear(num_clusters*cluster_dim+token_dim, hidden_size * 2)))
                else:
                    self.domain_prompt_mlp_list.append(nn.Sequential(nn.SiLU() if mlp_nonlinear else nn.Identity(),
                                                                nn.Linear(num_clusters*cluster_dim+token_dim, hidden_size * 6)))
                clusternorm1 = ClusterNorm(hidden_size)
                clusternorm1.norm.load_state_dict(blk.norm1.state_dict(), strict=False) # weight and bias
                clusternorm1.set_weight_bias(blk.norm1.weight, blk.norm1.bias)
                blk.norm1 = clusternorm1
                clusterls1 = ClusterLayerScale(hidden_size)
                clusterls1.set_gamma(blk.ls1.gamma)
                blk.ls1 = clusterls1
                if i == self.num_trainable_blocks - 1 and not self.final_layer_norm:
                    pass
                else:
                    clusternorm2 = ClusterNorm(hidden_size)
                    clusternorm2.norm.load_state_dict(blk.norm2.state_dict(), strict=False) # weight and bias
                    clusternorm2.set_weight_bias(blk.norm2.weight, blk.norm2.bias)
                    blk.norm2 = clusternorm2
                    clusterls2 = ClusterLayerScale(hidden_size)
                    clusterls2.set_gamma(blk.ls2.gamma)
                    blk.ls2 = clusterls2
            # Zero initialize the domain prompt mlp
            for domain_prompt_mlp in self.domain_prompt_mlp_list:
                nn.init.constant_(domain_prompt_mlp[1].weight, 0)
                nn.init.constant_(domain_prompt_mlp[1].bias, 0)
            # Zero the domain prompt mlp
            assert self.num_trainable_blocks > 0, 'First blocks should be frozen when using domain prompt'


    def forward(self, x, domain_idx=None):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        if self.num_trainable_blocks < 0:
            # All blocks are frozen
            assert domain_idx is None, 'Domain index should not be provided when all blocks are frozen'
            with torch.no_grad():
                for blk in self.model.blocks:
                    x = blk(x)
            x = x.detach()
        elif self.num_trainable_blocks == 0:
            # All blocks are trainable
            assert domain_idx is None or self.domain_prompt == False, 'Domain index should not be provided when all blocks are trainable or domain prompt is not used'
            for blk in self.model.blocks:
                x = blk(x)
        else:
            # First blocks are frozen
            with torch.no_grad():
                for blk in self.model.blocks[:-self.num_trainable_blocks]:
                    x = blk(x)
            x = x.detach()
            if self.domain_prompt:
                t = x[:, 0]
                f = x[:, 1:]
                # Reshape to (B, C, H, W)
                f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
                domain_prompt_desc = self.domain_prompt_model((f, t), domain_idx)
            # Last blocks are trained
            for i, blk in enumerate(self.model.blocks[-self.num_trainable_blocks:]):
                if self.domain_prompt:
                    domain_prompt_output = self.domain_prompt_mlp_list[i](domain_prompt_desc).chunk(6, dim=1)
                    blk.norm1.set_residual_weight_bias(domain_prompt_output[0], domain_prompt_output[1])
                    blk.ls1.set_residual_gamma(domain_prompt_output[2])
                    if i == self.num_trainable_blocks - 1 and not self.final_layer_norm:
                        pass
                    else:
                        blk.norm2.set_residual_weight_bias(domain_prompt_output[3], domain_prompt_output[4])
                        blk.ls2.set_residual_gamma(domain_prompt_output[5])
                x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

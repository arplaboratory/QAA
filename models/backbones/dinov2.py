import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from ..aggregators.salad import SALAD
from ..aggregators.queries_salad import QueriesSALAD
from ..aggregators.shared_queries_salad import SharedQueriesSALAD

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class QueryCrossAttnPrompt(torch.nn.Module):
    def __init__(self, in_dim, nheads=8):
        super(QueryCrossAttnPrompt, self).__init__()
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_in = torch.nn.LayerNorm(in_dim)

    def forward(self, x, q):
        out = self.norm_in(q)
        out, _ = self.cross_attn(q, x, x)
        return out
    
class ClusterNorm(nn.Module):
    def __init__(self, hidden_size, residual):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        if residual:
            self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "residual_weight") and hasattr(self, "residual_bias"):
            weight = self.weight + self.residual_weight
            bias = self.bias + self.residual_bias
        else:
            weight = self.weight
            bias = self.bias
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
        residual,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        if residual:
            self.gamma = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "residual_gamma"):
            gamma = self.gamma + self.residual_gamma
        else:
            gamma = self.gamma
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
            domain_prompt="none",
            injection_method="norm",
            injection_layer=0,
            num_clusters=16,
            cluster_dim=16,
            token_dim=256,
            self_attn=True,
            dust_bin=True,
            dropout=0.0,
            divide=1,
            divide_ratio=[1,1,1,0],
            residual=True,
            num_queries=64,
            multi_scale="1",
            multi_adapt="none",
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.domain_prompt = domain_prompt
        self.injection_method = injection_method
        self.injection_layer = len(self.model.blocks) - injection_layer
        self.residual = residual
        self.num_queries = num_queries
        self.multi_scale_layers = [len(self.model.blocks) - int(x) for x in multi_scale.split(",")]
        self.multi_adapt = multi_adapt
        
        if self.domain_prompt!="none":
            assert injection_layer > 0, 'Injection layer should be greater than 0'
            hidden_size = self.model.blocks[0].norm1.weight.shape[0]
            assert self.num_trainable_blocks > 0, 'First blocks should be frozen when using domain prompt'
            if self.domain_prompt == "QueriesSALAD":
                if multi_adapt == "none" or multi_adapt == "shared":
                    self.domain_prompt_model = QueriesSALAD(num_channels=hidden_size, num_clusters=num_clusters, cluster_dim=cluster_dim,
                                                    token_dim=token_dim, dropout=dropout,
                                                    divide=divide, divide_ratio=divide_ratio,
                                                    num_queries=num_queries)
                elif multi_adapt == "separate":
                    self.domain_prompt_model_list = nn.ModuleList()
                    for i, blk in enumerate(self.model.blocks[self.injection_layer:]):
                        self.domain_prompt_model_list.append(QueriesSALAD(num_channels=hidden_size, num_clusters=num_clusters, cluster_dim=cluster_dim,
                                                        token_dim=token_dim, dropout=dropout,
                                                        divide=divide, divide_ratio=divide_ratio,
                                                        num_queries=num_queries))
                else:
                    raise NotImplementedError()
            elif self.domain_prompt == "SharedQueriesSALAD":
                if multi_adapt == "none" or multi_adapt == "shared":
                    self.domain_prompt_model = SharedQueriesSALAD(num_channels=hidden_size, num_clusters=num_clusters, cluster_dim=cluster_dim,
                                                    token_dim=token_dim, dropout=dropout,
                                                    divide=divide, divide_ratio=divide_ratio,
                                                    num_queries=num_queries)
                elif multi_adapt == "separate":
                    self.domain_prompt_model_list = nn.ModuleList()
                    for i, blk in enumerate(self.model.blocks[self.injection_layer:]):
                        self.domain_prompt_model_list.append(SharedQueriesSALAD(num_channels=hidden_size, num_clusters=num_clusters, cluster_dim=cluster_dim,
                                                        token_dim=token_dim, dropout=dropout,
                                                        divide=divide, divide_ratio=divide_ratio,
                                                        num_queries=num_queries))
                else:
                    raise NotImplementedError()
            else:
                raise ValueError(f'Unknown domain prompt {self.domain_prompt}')
            if self.injection_method == "norm":
                self.domain_prompt_mlp_list = nn.ModuleList()
                for i, blk in enumerate(self.model.blocks[self.injection_layer:]):
                    self.domain_prompt_mlp_list.append(nn.Sequential(nn.Linear(num_clusters*cluster_dim+token_dim, hidden_size * 6),
                                                                    nn.SiLU(),
                                                                    nn.Linear(hidden_size * 6, hidden_size * 6)
                                                                    ))
                    clusternorm1 = ClusterNorm(hidden_size, residual)
                    clusternorm1.norm.load_state_dict(blk.norm1.state_dict(), strict=False)
                    if self.residual:
                        clusternorm1.set_weight_bias(blk.norm1.weight, blk.norm1.bias)
                    blk.norm1 = clusternorm1
                    clusterls1 = ClusterLayerScale(hidden_size, residual)
                    if self.residual:
                        clusterls1.set_gamma(blk.ls1.gamma)
                    blk.ls1 = clusterls1
                    clusternorm2 = ClusterNorm(hidden_size, residual)
                    clusternorm2.norm.load_state_dict(blk.norm2.state_dict(), strict=False)
                    if self.residual:
                        clusternorm2.set_weight_bias(blk.norm2.weight, blk.norm2.bias)
                    blk.norm2 = clusternorm2
                    clusterls2 = ClusterLayerScale(hidden_size, residual)
                    if self.residual:
                        clusterls2.set_gamma(blk.ls2.gamma)
                    blk.ls2 = clusterls2
                # Zero initialize the domain prompt mlp
                for domain_prompt_mlp in self.domain_prompt_mlp_list:
                    nn.init.constant_(domain_prompt_mlp[0].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[0].bias, 0)
                    nn.init.constant_(domain_prompt_mlp[2].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[2].bias, 0)
            elif self.injection_method == "add":
                self.domain_prompt_mlp_list = nn.ModuleList()
                for i, blk in enumerate(self.model.blocks[self.injection_layer:]):
                    self.domain_prompt_mlp_list.append(nn.Sequential(nn.Linear(num_clusters*cluster_dim+token_dim, hidden_size),
                                                                    nn.SiLU(),
                                                                    nn.Linear(hidden_size, hidden_size)
                                                                    ))
                # Zero initialize the domain Fprompt mlp
                for domain_prompt_mlp in self.domain_prompt_mlp_list:
                    nn.init.constant_(domain_prompt_mlp[0].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[0].bias, 0)
                    nn.init.constant_(domain_prompt_mlp[2].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[2].bias, 0)
            elif self.injection_method == "add_adapter":
                self.shared_prompt_mlp = nn.Sequential(nn.Linear(num_clusters*cluster_dim+token_dim, hidden_size),
                                                        nn.SiLU(),
                                                        nn.Linear(hidden_size, hidden_size))
                self.domain_prompt_mlp_list = nn.ModuleList()
                for i, blk in enumerate(self.model.blocks[self.injection_layer:]):
                    self.domain_prompt_mlp_list.append(nn.Sequential(nn.Linear(hidden_size, hidden_size // 4),
                                                                    nn.SiLU(),
                                                                    nn.Linear(hidden_size // 4, hidden_size)
                                                                    )) # Adapter
                # Zero initialize the domain prompt mlp
                nn.init.constant_(self.shared_prompt_mlp[0].weight, 0)
                nn.init.constant_(self.shared_prompt_mlp[0].bias, 0)
                nn.init.constant_(self.shared_prompt_mlp[2].weight, 0)
                nn.init.constant_(self.shared_prompt_mlp[2].bias, 0)
                for domain_prompt_mlp in self.domain_prompt_mlp_list:
                    nn.init.constant_(domain_prompt_mlp[0].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[0].bias, 0)
                    nn.init.constant_(domain_prompt_mlp[2].weight, 0)
                    nn.init.constant_(domain_prompt_mlp[2].bias, 0)
            else:
                raise ValueError(f'Unknown injection method {self.injection_method}')


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
        
        feature_list = []
        layer_count = 0
        domain_prompt_desc = None
        if self.domain_prompt != "none" and self.multi_adapt != "none":
            domain_prompt_desc_list = []
        # First blocks are frozen
        for i, blk in enumerate(self.model.blocks):
            # print(f"Before {i} block")
            if self.domain_prompt != "none":
                if self.multi_adapt == "none" and layer_count == self.injection_layer:
                    t = x[:, 0]
                    f = x[:, 1:]
                    # Reshape to (B, C, H, W)
                    f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
                    domain_prompt_desc = self.domain_prompt_model((f, t), domain_idx)
                elif (self.multi_adapt == "shared" or self.multi_adapt == "separate") and layer_count >= self.injection_layer:
                    t = x[:, 0]
                    f = x[:, 1:]
                    # Reshape to (B, C, H, W)
                    f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
                    if self.multi_adapt == "shared":
                        domain_prompt_desc = self.domain_prompt_model((f, t), domain_idx)
                        domain_prompt_desc_list.append(domain_prompt_desc)
                    elif self.multi_adapt == "separate":
                        domain_prompt_desc = self.domain_prompt_model_list[i - self.injection_layer]((f, t), domain_idx)
                        domain_prompt_desc_list.append(domain_prompt_desc)
                # print(f"Use feature after block {i-1} to output domain prompt desc")
            if domain_prompt_desc is not None:
                # print(f"Use domain prompt desc and domain mlp {i-self.injection_layer} for block {i}")
                if self.injection_method == "norm":
                    domain_prompt_output = self.domain_prompt_mlp_list[i - self.injection_layer](domain_prompt_desc).chunk(6, dim=1)
                    if self.residual:
                        blk.norm1.set_residual_weight_bias(domain_prompt_output[0], domain_prompt_output[1])
                        blk.ls1.set_residual_gamma(domain_prompt_output[2])
                        blk.norm2.set_residual_weight_bias(domain_prompt_output[3], domain_prompt_output[4])
                        blk.ls2.set_residual_gamma(domain_prompt_output[5])
                    else:
                        blk.norm1.set_weight_bias(domain_prompt_output[0], domain_prompt_output[1])
                        blk.ls1.set_gamma(domain_prompt_output[2])
                        blk.norm2.set_weight_bias(domain_prompt_output[3], domain_prompt_output[4])
                        blk.ls2.set_gamma(domain_prompt_output[5])
                    x = blk(x)
                elif self.injection_method == "add":
                    x = blk(x + self.domain_prompt_mlp_list[i - self.injection_layer](domain_prompt_desc).unsqueeze(1)) # domain_prompt_output broadcasting
                elif self.injection_method == "add_adapter":
                    domain_prompt_output = self.shared_prompt_mlp(domain_prompt_desc)
                    x = blk(x + self.domain_prompt_mlp_list[i - self.injection_layer](domain_prompt_output).unsqueeze(1)) # domain_prompt_output broadcasting
                else:
                    raise ValueError(f'Unknown injection method {self.injection_method}')
                # print(f"After {i} block")
            else:
                x = blk(x)
                # print(f"After {i} block")
            if layer_count in self.multi_scale_layers:
                # print(f"Featured {i} add")
                feature_list.append(x)
            layer_count+=1
            if layer_count > self.multi_scale_layers[0]: # Break if we have all the multi_scale layers
                break

        for i in range(len(feature_list)):
            if i != 0:
                if self.norm_layer:
                    x = self.model.norm(feature_list[i])
                
                t_out = x[:, 0] # Only use last token
                # t_out = torch.cat([t_out, t], dim=1) # Use all tokens
                f = x[:, 1:]

                # Reshape to (B, C, H, W)
                f_out = torch.cat([f_out, f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)], dim=2)
            else:
                if self.norm_layer:
                    x = self.model.norm(x)
                
                t_out = x[:, 0]
                f = x[:, 1:]

                # Reshape to (B, C, H, W)
                f_out = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            if self.domain_prompt != "none" and self.multi_adapt != "none":
                return f_out, t_out, domain_prompt_desc_list
            elif self.domain_prompt != "none":
                return f_out, t_out, domain_prompt_desc
            return f_out, t_out
        return f_out

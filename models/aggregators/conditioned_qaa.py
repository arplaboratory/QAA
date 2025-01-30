import torch
import torch.nn as nn
from .attention import QuerySelfAttn, QueryCrossAttn
from .salad import log_optimal_transport


class ConditionedQAA(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
            divide_ratio=[1,1,1,0],
            divide=1,
            num_queries=32,
            self_attn="both",
            dust_bin=True,
            freeze="none",
            feature_nheads=4,
            score_nheads=4,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.freeze = freeze
        self.divide = divide
        self.divide_ratio = divide_ratio
        if self.divide > 1:
            raise NotImplementedError()
        self.num_queries = num_queries
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        if self.token_dim != 0:
            self.token_features = nn.Sequential(
                nn.Linear(self.num_channels, 512),
                nn.ReLU(),
                nn.Linear(512, self.token_dim)
            )
        # MLP for local features f_i
        # MLP for score matrix S
        if self_attn == "both" or self_attn == "score":
            self.queries_score = QuerySelfAttn(self.num_clusters, self.num_queries, nheads=score_nheads, self_attn=True)
        else:
            self.queries_score = QuerySelfAttn(self.num_clusters, self.num_queries, nheads=score_nheads, self_attn=False)
        if self_attn == "both" or self_attn == "feature":
            self.queries_feature = QuerySelfAttn(self.cluster_dim, self.num_queries, nheads=feature_nheads, self_attn=True)
        else:
            self.queries_feature = QuerySelfAttn(self.cluster_dim, self.num_queries, nheads=feature_nheads, self_attn=False)
        self.score = QueryCrossAttn(self.num_clusters, self.num_clusters, nheads=score_nheads)
        self.cluster_feature = QueryCrossAttn(self.cluster_dim, self.cluster_dim, nheads=feature_nheads)
        self.proj_s = torch.nn.Conv2d(self.num_channels, self.num_clusters, kernel_size=3, padding=1)
        self.proj_f = torch.nn.Conv2d(self.num_channels, self.cluster_dim, kernel_size=3, padding=1)
        if divide > 1:
            raise NotImplementedError()
        # Dustbin parameter z
        if dust_bin:
            self.dust_bin = nn.Parameter(torch.tensor(1.))
        else:
            self.dust_bin = None


    def forward(self, x, domain_idx=None, visualize=False):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].
        domain_idx (torch.Tensor, optional): The domain index tensor [B]. Defaults to None.

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        if len(x) == 3:
            x, t, domain_desc = x
        else:
            x, t = x # Extract features and token
            domain_desc = None
        x_s = self.proj_s(x)
        x_f = self.proj_f(x)
        f = self.queries_feature().repeat(x.shape[0], 1, 1)
        q = self.queries_score().repeat(x.shape[0], 1, 1)
        f, f_attn = self.cluster_feature(x_f, f)
        p, p_attn = self.score(x_s, q)
        if self.divide > 1:
            raise NotImplementedError()
        if self.token_dim != 0:
            t = self.token_features(t)
        assert p.shape[1] == self.num_clusters
        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        if self.dust_bin is not None:
            p = p[:, :-1, :]


        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        if self.token_dim == 0:
            f = nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        else:
            f = torch.cat([
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
            ], dim=-1)

        if domain_desc is not None:
            return nn.functional.normalize(f, p=2, dim=-1), domain_desc
        if visualize:
            return nn.functional.normalize(f, p=2, dim=-1), p_attn, p
        return nn.functional.normalize(f, p=2, dim=-1)
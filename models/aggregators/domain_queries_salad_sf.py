import torch
import torch.nn as nn
from .attention import QuerySelfAttn, QueryCrossAttn
from .salad import log_optimal_transport


class DomainQueriesSALADSF(nn.Module):
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
            self_attn=True,
            dust_bin=True,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.divide = divide
        self.divide_ratio = divide_ratio
        if self.divide > 1:
            assert self.divide == len(self.divide_ratio) - 1 # Last one for shared clusters
            assert num_clusters % sum(self.divide_ratio) == 0
            assert num_queries % sum(self.divide_ratio) == 0
            self.divide_query_list = [num_queries * self.divide_ratio[i] // sum(self.divide_ratio)  for i in range(len(divide_ratio))]
            self.divide_query_idx_list = [sum(self.divide_query_list[:i+1]) for i in range(len(self.divide_query_list))]
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
        if divide > 1:
            self.queries_score_list = nn.ModuleList([
                QuerySelfAttn(self.num_channels, divide_queries, nheads=self.num_channels // 64, self_attn=self_attn) if divide_queries != 0 else None for divide_queries in self.divide_query_list
            ])
            self.queries_feature_list = nn.ModuleList([
                QuerySelfAttn(self.cluster_dim, divide_queries, nheads=self.cluster_dim // 32, self_attn=self_attn) if divide_queries != 0 else None for divide_queries in self.divide_query_list
            ])
            self.score = QueryCrossAttn(self.num_channels, self.num_clusters, nheads=self.num_channels // 64)
        else:
            raise NotImplementedError()
        # Dustbin parameter z
        if dust_bin:
            self.dust_bin = nn.Parameter(torch.tensor(1.))
        else:
            self.dust_bin = None


    def forward(self, x, domain_idx=None):
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

        if self.divide > 1:
            # Use decoupled score network
            if domain_idx is None:
                f = self.queries_feature_list().permute(0, 2, 1).repeat(x.shape[0], 1, 1)
                p = self.score(x, self.queries_score_list()[0].repeat(x.shape[0], 1, 1))[0]
            else:
                f = self.generate_score_from_decoupled_fnet(x, self.queries_feature_list, domain_idx, "feature")
                p = self.generate_score_from_decoupled_fnet(x, self.queries_score_list, domain_idx, "score")
        else:
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
        return nn.functional.normalize(f, p=2, dim=-1)
    
    def generate_score_from_decoupled_fnet(self, x, q, domain_idx, type=None):
        if type == "feature":
            q_f, q_f_detach = q()
            q_f = q_f.permute(0, 2, 1)
            q_f_detach = q_f_detach.permute(0, 2, 1)
            q_f_new = []
            for i in range(len(self.divide_query_list)): # For each domain
                if self.divide_query_list[i] > 0:
                    q_f_list = []
                    for j in torch.unique(domain_idx):
                        bs = (domain_idx == j).sum()
                        if j == 0:
                            start = 0
                            end = self.divide_query_idx_list[j]
                        else:
                            start = self.divide_query_idx_list[j-1]
                            end = self.divide_query_idx_list[j]
                        if i == j:
                            domain_q_f = q_f[:, :, start : end].repeat(bs, 1, 1)
                        else:
                            domain_q_f = q_f_detach[:, :, start : end].repeat(bs, 1, 1)
                        q_f_list.append(domain_q_f)
                    q_f_new.append(torch.cat(q_f_list, dim=0))
            q_f_new = torch.cat(q_f_new, dim=2)
            return q_f_new
        elif type == "score":
            q_f, q_f_detach = q()
            q_f_new = []
            for i in range(len(self.divide_query_list)): # For each domain
                if self.divide_query_list[i] > 0:
                    q_f_list = []
                    for j in torch.unique(domain_idx):
                        bs = (domain_idx == j).sum()
                        if j == 0:
                            start = 0
                            end = self.divide_query_idx_list[j]
                        else:
                            start = self.divide_query_idx_list[j-1]
                            end = self.divide_query_idx_list[j]
                        if i == j:
                            domain_q_f = q_f[:, start : end].repeat(bs, 1, 1)
                        else:
                            domain_q_f = q_f_detach[:, start : end].repeat(bs, 1, 1)
                        q_f_list.append(domain_q_f)
                    q_f_new.append(torch.cat(q_f_list, dim=0))
            q_f_new = torch.cat(q_f_new, dim=1)
            f = self.score(x, q_f_new)[0]
            return f
        else:
            raise NotImplementedError()
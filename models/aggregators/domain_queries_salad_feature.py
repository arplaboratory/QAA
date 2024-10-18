import torch
import torch.nn as nn
from .attention import QuerySelfAttn, QueryCrossAttn

# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m*one).to(scores), (n*one).to(scores), ((n-m)*one).to(scores)

    if alpha is not None:
        bins = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)
        
        couplings = torch.cat([scores, bins], 1)
    else:
        couplings = scores

    norm = - (ms + ns).log()
    if alpha is not None:
        log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    else:
        log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class DomainQueriesSALADFeature(nn.Module):
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
            self.divide_cluster_list = [num_clusters  for i in range(len(divide_ratio))]
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
            self.queries_feature_list = nn.ModuleList([
                QuerySelfAttn(self.num_channels, divide_queries, nheads=self.num_channels // 64, self_attn=self_attn) if divide_queries != 0 else None for divide_queries in self.divide_query_list
            ])
            self.queries_score = QuerySelfAttn(self.num_channels, self.num_queries, nheads=self.num_channels // 64, self_attn=self_attn)
            self.cluster_features = QueryCrossAttn(self.num_channels, self.cluster_dim, nheads=self.num_channels // 64)
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

        q = self.queries_score()
        if self.divide > 1:
            # Use decoupled score network
            if domain_idx is None:
                f_list = [self.cluster_features(x, self.queries_feature_list[i]())[0] if self.divide_query_list[i] != 0 else None for i in range(len(self.divide_query_list))]
                f_list = [f for f in f_list if f is not None]
                f = torch.cat(f_list, dim=2) # For each domain
            else:
                f = self.generate_score_from_decoupled_fnet(x, self.queries_feature_list, domain_idx, self.cluster_features)
            p = self.score(x, q)[0]
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
    
    def generate_score_from_decoupled_fnet(self, x, q, domain_idx, network):
        f_list = [network(x, q[i]())[0] if self.divide_query_list[i] != 0 else None for i in range(len(self.divide_query_list))]
        f_list_new = []
        for i in range(len(self.divide_cluster_list)): # For each domain
            if f_list[i] is not None:
                f_list_single = []
                for j in torch.unique(domain_idx):
                    domain_f = f_list[i][domain_idx == j]
                    if len(domain_f) == 0:
                        continue
                    if j != i:
                        domain_f = domain_f.detach()
                    f_list_single.append(domain_f)
                f_list_new.append(torch.cat(f_list_single, dim=0))
        f = torch.cat(f_list_new, dim=2)
        return f
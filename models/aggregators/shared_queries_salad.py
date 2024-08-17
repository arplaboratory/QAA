import torch
import torch.nn as nn

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

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    
    couplings = torch.cat([scores, bins], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class QuerySelfAttn(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(QuerySelfAttn, self).__init__()
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

    def forward(self):
        # B = x.size(0)

        # q = self.queries.repeat(B, 1, 1)
        q = self.queries
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        return q
        
class QueryCrossAttn(torch.nn.Module):
    def __init__(self, in_dim, output_dim, nheads=8):
        super(QueryCrossAttn, self).__init__()
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.conv = torch.nn.Conv1d(in_dim, output_dim, 1)

    def forward(self, x, q):
        B = x.size(0)

        q = q.repeat(B, 1, 1)
        x_flatten = x.flatten(2).permute(0, 2, 1)
        
        out, attn = self.cross_attn(q, x_flatten, x_flatten)
        out = self.norm_out(out)
        out = self.conv(out.permute(0, 2, 1))
        return out, attn

class SharedQueriesSALAD(nn.Module):
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
            divide=1,
            shared_clusters=0,
            padding="detach",
            num_queries=32,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.divide = divide
        if divide > 1:
            self.shared_clusters = shared_clusters
            self.specific_clusters = (self.num_clusters - shared_clusters) // divide
        self.padding = padding # Ensure the dimension is the same
        self.num_queries = num_queries
        assert self.padding in ["detach", "none"]
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        if divide > 1:
            self.queries = QuerySelfAttn(self.num_channels, self.num_queries, nheads=self.num_channels // 64)
            # MLP for local features f_i
            self.cluster_features = QueryCrossAttn(self.num_channels, self.cluster_dim, nheads=self.num_channels // 64)
            if self.shared_clusters > 0:
                self.shared_score = QueryCrossAttn(self.num_channels, self.shared_clusters, nheads=self.num_channels // 64)
            else:
                self.shared_score = None
            self.score_list = nn.ModuleList([
                 QueryCrossAttn(self.num_channels, self.specific_clusters, nheads=self.num_channels // 64) for _ in range(divide)
            ])
        else:
            self.queries = QuerySelfAttn(self.num_channels, self.num_queries, nheads=self.num_channels // 64)
            # MLP for local features f_i
            self.cluster_features = QueryCrossAttn(self.num_channels, self.cluster_dim, nheads=self.num_channels // 64)
            # MLP for score matrix S
            self.score =  QueryCrossAttn(self.num_channels, self.num_clusters, nheads=self.num_channels // 64)
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x, domain_idx=None):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].
        domain_idx (torch.Tensor, optional): The domain index tensor [B]. Defaults to None.

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x # Extract features and token

        q = self.queries()
        f, f_attn = self.cluster_features(x, q)
        if self.divide > 1:
            # Use decoupled score network
            if domain_idx is None:
                if self.shared_clusters > 0:
                    p_shared = self.shared_score(x, q)[0]
                p = torch.cat([self.score_list[i](x, q)[0] for i in range(self.divide)], dim=1) # For each domain
                if self.shared_clusters > 0:
                    p = torch.cat([p_shared, p], dim=1)
            else:
                if self.shared_clusters > 0:
                    p_shared = self.shared_score(x, q)[0]
                p = self.generate_score_from_decoupled_pnet(x, q, domain_idx)
                if self.shared_clusters > 0:
                    p = torch.cat([p_shared, p], dim=1)
        else:
            p, p_attn = self.score(x, q)
        t = self.token_features(t)
        assert p.shape[1] == self.num_clusters if self.padding in ["detach"] else self.shared_clusters + self.specific_clusters
        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]


        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        if self.padding in ["detach"] or domain_idx is None:
            f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        else:
            f = f.unsqueeze(2).repeat(1, 1, self.shared_clusters + self.specific_clusters, 1)

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)

    def pad_zero_score(self, p, domain_idx):
        p_zero = torch.zeros((p.shape[0], self.num_clusters, p.shape[2]), device=p.device)
        for i, domain_id_single in enumerate(domain_idx):
            p_zero[i, domain_id_single * self.specific_clusters: (domain_id_single + 1) * self.specific_clusters] = p[i]
        return p_zero
    
    def generate_score_from_decoupled_pnet(self, x, q, domain_idx):
        if self.padding == "none":
            p = torch.cat([self.score_list[i](x[domain_idx == i], q)[0] for i in range(self.divide)], dim=0)
        elif self.padding == "detach":
            p_list = [self.score_list[i](x, q)[0] for i in range(self.divide)]
            for i in range(self.divide): # For each domain
                p_list[i][domain_idx != i] = p_list[i][domain_idx != i].detach() # detach the other domains
            p = torch.cat(p_list, dim=1)
        return p
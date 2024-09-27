import torch
import torch.nn as nn

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

class QueriesAttention(nn.Module):
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
            cluster_dim=128,
            token_dim=256,
            divide=1,
            num_queries=32,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.divide = divide
        self.num_queries = num_queries
        
        self.queries = QuerySelfAttn(self.num_channels, self.num_queries, nheads=self.num_channels // 64)
        self.cluster_features =  QueryCrossAttn(self.num_channels, self.cluster_dim, nheads=self.num_channels // 64)
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )

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
        t = self.token_features(t)

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize(f.flatten(1), p=2, dim=1).flatten(1)
            ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)
import torch
import torch.nn as nn

class QuerySelfAttn(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, self_attn=True):
        super(QuerySelfAttn, self).__init__()
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = self_attn
        
        if self.self_attn:
            # the following two lines are used during training only, you can cache their output in eval.
            self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

    def forward(self, detach=False):
        # B = x.size(0)

        # q = self.queries.repeat(B, 1, 1)
        if detach:
            q, q_detach = self.queries, self.queries.detach()
            if self.self_attn:
                # the following two lines are used during training.
                # for stability purposes 
                q = q + self.self_attn(q, q, q)[0]
                q_detach = q_detach + self.self_attn(q_detach, q_detach, q_detach)[0]
            q = self.norm_q(q)
            q_detach = self.norm_q(q_detach)
            #######
            
            return q, q_detach
        else:
            q = self.queries
            if self.self_attn:
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

    def forward(self, x, q):
        x_flatten = x.flatten(2).permute(0, 2, 1)
        
        out, attn = self.cross_attn(q, x_flatten, x_flatten)
        out = self.norm_out(out).permute(0, 2, 1)
        return out, attn
import torch
import torch.nn as nn

class QuerySelfAttn(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, self_attn=True, detach=False):
        super(QuerySelfAttn, self).__init__()
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = self_attn
        self.detach = detach
        
        if self.self_attn:
            # the following two lines are used during training only, you can cache their output in eval.
            self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
            self.norm_q = torch.nn.LayerNorm(in_dim)
            #####

    def forward(self):
        # B = x.size(0)

        # q = self.queries.repeat(B, 1, 1)
        q = self.queries
        if self.self_attn:
            # the following two lines are used during training.
            # for stability purposes 
            q = q + self.self_attn(q, q, q)[0]
            q = self.norm_q(q)
        #######
        
        if self.detach:
            return q, q.detach()
        return q

class QueryCrossAttn(torch.nn.Module):
    def __init__(self, in_dim, output_dim, nheads=8):
        super(QueryCrossAttn, self).__init__()
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.norm_out2 = torch.nn.LayerNorm(output_dim)
        self.conv = torch.nn.Conv1d(in_dim, output_dim, 1)

    def forward(self, x, q):
        x_flatten = x.flatten(2).permute(0, 2, 1)
        
        out, attn = self.cross_attn(q, x_flatten, x_flatten)
        out = self.norm_out(out)
        out = self.conv(out.permute(0, 2, 1))
        out = self.norm_out2(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out, attn
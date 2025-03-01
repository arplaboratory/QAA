import torch
from torch import nn
class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        
    def cache_query(self):
        q = self.queries
        self.cached_q = q + self.self_attn(q, q, q)[0]
        self.cached_q = self.norm_q(self.cached_q)

    def clean_cache(self):
        del self.cached_q

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        if hasattr(self, "cached_q"):
            q = self.cached_q
            q = q.repeat(B, 1, 1)
        else:
            q = self.queries.repeat(B, 1, 1)
            
            # the following two lines are used during training.
            # for stability purposes 
            q = q + self.self_attn(q, q, q)[0]
            q = self.norm_q(q)
            #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=768, proj_channels=384, num_queries=64, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def cache_query(self):
        for i in range(len(self.boqs)):
            self.boqs[i].cache_query()

    def clean_cache(self):
        for i in range(len(self.boqs)):
            self.boqs[i].clean_cache()

    def forward(self, x, domain_idx=None, visualize=None):
        # reduce input dimension using 3x3 conv when using ResNet
        if len(x) == 3:
            x, t, domain_desc = x
        else:
            x, t = x # Extract features and token
            domain_desc = None
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        if domain_desc is not None:
            return nn.functional.normalize(out, p=2, dim=-1), domain_desc
        if visualize:
            raise NotImplementedError()
        return nn.functional.normalize(out, p=2, dim=-1)
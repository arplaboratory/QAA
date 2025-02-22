import torch
import torch.nn as nn

class QuerySelfAttn(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, self_attn_flag=True, self_attn_out_norm=True):
        super(QuerySelfAttn, self).__init__()
        
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn_flag = self_attn_flag
        self.self_attn_out_norm = self_attn_out_norm
        
        if self.self_attn_flag:
            # the following two lines are used during training only, you can cache their output in eval.
            self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

    def adjust_queries(self, num_queries):
        if num_queries > self.queries.shape[1]:
            expand_ratio = num_queries // self.queries.shape[1] + 1
            new_queries = self.queries.repeat(1, expand_ratio, 1)
            new_queries = new_queries[:, :num_queries, :]
        else:
            new_queries = self.queries[:, :num_queries, :]
        self.queries = nn.Parameter(new_queries)

    def forward(self, detach=False):
        # B = x.size(0)

        # q = self.queries.repeat(B, 1, 1)
        if detach:
            q, q_detach = self.queries, self.queries.detach()
            if self.self_attn_flag:
                # the following two lines are used during training.
                # for stability purposes 
                q = q + self.self_attn(q, q, q)[0]
                q_detach = q_detach + self.self_attn(q_detach, q_detach, q_detach)[0]
            if self.self_attn_out_norm:
                q = self.norm_q(q)
                q_detach = self.norm_q(q_detach)
            #######
            
            return q, q_detach
        else:
            q = self.queries
            if self.self_attn_flag:
                # the following two lines are used during training.
                # for stability purposes 
                q = q + self.self_attn(q, q, q)[0]
            if self.self_attn_out_norm:
                print("YES2")
                q = self.norm_q(q)
            #######
            
            return q
        
class QueryCrossAttn(torch.nn.Module):
    def __init__(self, in_dim, output_dim, nheads=8, arch="conv", skip="none", out_norm=True):
        super(QueryCrossAttn, self).__init__()
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        self.arch = arch
        self.skip = skip
        self.out_norm = out_norm
        if self.arch == "conv":
            self.conv = torch.nn.Conv1d(in_dim, output_dim, 1)
            self.norm2_out = torch.nn.LayerNorm(output_dim)
        elif self.arch == "linear":
            self.linear1 = torch.nn.Linear(in_dim, 4*in_dim, bias=True)
            self.activation = torch.nn.ReLU(inplace=True)
            self.linear2 = torch.nn.Linear(4*in_dim, output_dim, bias=True)
            self.norm2_out = torch.nn.LayerNorm(output_dim)
        elif self.arch == "linearproj":
            self.linear1 = torch.nn.Linear(in_dim, 4*in_dim, bias=True)
            self.activation = torch.nn.ReLU(inplace=True)
            self.linear2 = torch.nn.Linear(4*in_dim, in_dim, bias=True)
            self.norm2_out = torch.nn.LayerNorm(in_dim)
            self.linear3 = torch.nn.Linear(in_dim, output_dim, bias=True)
            self.norm3_out = torch.nn.LayerNorm(output_dim)
        elif self.arch == "none":
            pass
        else:
            raise NotImplementedError()

    def forward(self, x, q):
        x_flatten = x.flatten(2).permute(0, 2, 1)
        
        out, attn = self.cross_attn(q, x_flatten, x_flatten)
        if self.skip == "full" or self.skip == "cross":
            out = q + out
        out = self.norm_out(out)
        if self.arch == "conv":
            cache = out
            out = self.conv(out.permute(0, 2, 1)).permute(0, 2, 1)
            if self.skip == "full":
                out = cache + out
            if self.out_norm:
                print("YES1")
                out = self.norm2_out(out)
        elif self.arch == "linear":
            cache = out
            out = self.linear2(self.activation(self.linear1(out)))
            if self.skip == "full":
                out = cache + out
            if self.out_norm:
                out = self.norm2_out(out)
        elif self.arch == "linearproj":
            cache = out
            out = self.linear2(self.activation(self.linear1(out)))
            if self.skip == "full":
                out = cache + out
            out = self.norm2_out(out)
            out = self.linear3(out)
            if self.out_norm:
                out = self.norm3_out(out)
        elif self.arch == "none": # Only use when the dim is projected
            pass
        out = out.permute(0, 2, 1)
        return out, attn
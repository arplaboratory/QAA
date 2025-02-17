from lightning.fabric.utilities.throughput import measure_flops
import torch

def measure_flop(model):
    model = model.cuda()
    x = torch.randn(1, 3, 322, 322).cuda()
    if model.agg_arch == "QAA":
        model.aggregator.cache_query()
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)/ 1e9
    model_backbone_fwd = lambda: model.backbone(x)
    backbone_flops = measure_flops(model, model_backbone_fwd)/ 1e9
    model.logger.experiment.log({"fwd_flops": fwd_flops,
                                 "backbone_flops": backbone_flops,
                                 "agg_flops": fwd_flops-backbone_flops,
                                 "model_output_dim": model(x).shape[1]})
    print(f"Model: {fwd_flops} GFLOPS, Backbone: {backbone_flops}, Agg: {fwd_flops-backbone_flops}")
    print(f"Model dim: {model(x).shape}")
    if model.agg_arch == "QAA":
        model.aggregator.clean_cache()
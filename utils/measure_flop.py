from lightning.fabric.utilities.throughput import measure_flops
import torch

def measure_flop(model):
    model = model.cuda()
    x = torch.randn(1, 3, 322, 322).cuda()
    if model.agg_arch == "QAA" or model.agg_arch == "BoQ":
        print(model.aggregator)
        model.aggregator.cache_query()
    model_fwd = lambda: model(x)
    fwd_flops = measure_flops(model, model_fwd)/ 1e9
    model_backbone_fwd = lambda: model.backbone(x)
    backbone_flops = measure_flops(model, model_backbone_fwd)/ 1e9
    try:
        model_output_dim = model(x).shape[1]
    except Exception as e:
        model_output_dim = 0
    model.logger.experiment.log({"fwd_flops": fwd_flops,
                                 "backbone_flops": backbone_flops,
                                 "agg_flops": fwd_flops-backbone_flops,
                                 "model_output_dim": model_output_dim})
    print(f"Model: {fwd_flops} GFLOPS, Backbone: {backbone_flops}, Agg: {fwd_flops-backbone_flops}")
    if model.agg_arch == "QAA" or model.agg_arch == "BoQ":
        model.aggregator.clean_cache()
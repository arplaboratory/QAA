import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import argparse
from torchvision.transforms import v2
from torchvision import transforms as T

from models import helper

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

test_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize([322, 322], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])])

class QAA(nn.Module, PyTorchModelHubMixin):
    """
    Hugging Face compatible version of the VPRModel.
    Focuses on inference and modular architecture loading.
    """
    def __init__(self, model_config_dict):
        super().__init__()
        
        # Store model_config_dict for serializability
        self.model_config = model_config_dict
        
        # Extract architecture settings from config
        backbone_arch = model_config_dict['backbone_arch']
        backbone_config = model_config_dict['backbone_config']
        agg_arch =  model_config_dict['agg_arch']
        agg_config =  model_config_dict['agg_config']
        
        # Decorrelation settings
        self.decorrelation = 'none'

        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
            
        self.agg_arch = agg_arch
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    def forward(self, x, domain_idx=None):
        """
        Forward pass for global descriptor extraction.
        """
        # Feature Extraction
        x = self.backbone(x, domain_idx=domain_idx)
        
        # Feature Aggregation
        # Handling different return types based on decorrelation settings
        if self.decorrelation == "none" or domain_idx is None:
            # Standard inference: returns global descriptor
            x = self.aggregator(x, domain_idx=domain_idx)
        else:
            # Training-time style forward: returns (descriptor, query_p, query_f)
            x = self.aggregator(x, domain_idx=domain_idx, decorrelation=self.decorrelation)
            
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load directly from Hugging Face Hub
    print(f"Loading model on {device}...")
    model = QAA.from_pretrained("xjh19972/QAA-1024").to(device) # QAA-8192, QAA-4096, QAA-2048, QAA-1024
    model.eval()

    # 2. Run Inference
    image = torch.randn(1, 3, 322, 322).to(device)
    
    with torch.no_grad():
        input_tensor = test_transform(image)
        descriptor = model(input_tensor)
    
    print(f"Success. Descriptor shape: {descriptor.shape}")
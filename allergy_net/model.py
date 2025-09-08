import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=87):
        super(ViTClassifier, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Check type of heads
        if isinstance(self.vit.heads, nn.Linear):
            in_features = self.vit.heads.in_features
            self.vit.heads = nn.Linear(in_features, num_classes)
        elif isinstance(self.vit.heads, nn.Sequential):
            # Assuming last layer is Linear
            in_features = self.vit.heads[-1].in_features
            self.vit.heads[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unknown ViT head type")

    def forward(self, x):
        return self.vit(x)

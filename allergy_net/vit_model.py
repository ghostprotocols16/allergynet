import torch
import torch.nn as nn
import timm

def get_vit_model(num_classes=5000, pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    # Replace classifier head
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

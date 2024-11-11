import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Functional as F

class image_encoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048,embed_dim)
    def forward(self, x):
        return self.resnet(x)
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

class text_encoder(nn.Module):
    def __init__(self, vocab_size=50, context_length=64, transformer_width=128 ):
        super().__init__()
        self.vocab_size = vocab_size # We can design only 50 vocab size because we are going to only give in parameters and some text and also some special characters
        self.context_length = context_length # set to 64, as we are gong to process at max 10 params, we can increase this value in future based on need
        self.transformer_width = transformer_width # internal feature size of the transformer, represents each token's hidden representation
        
class MultiheadAttention(nn.Module):
    def __init__(self,d_model, heads):
        super().__init__()
        self.d_model = d_model #d_model as mentioned in paper
        self.heads = heads # num transformer heads
        self.scale = (self.d_model/self.heads)**-0.5 # the denominator sq.(dv) in paper is modified as scale
    def forward(self,x):
        pass    




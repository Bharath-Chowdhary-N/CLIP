import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CLIP import CLIP

def main():

    model = CLIP(embed_dim=128, temperature=0.1)
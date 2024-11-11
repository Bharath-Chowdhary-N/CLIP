import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CLIP import CLIP

def train_clip(model):
    #add code to train the model
    pass

def main():

    model = CLIP(embed_dim=128, temperature=0.1)

    #add code to make image text pairs

    #add code to train model
    trained_model = train_clip(model=model)

    #save the trained model

if __name__ == "__main__":
    main()
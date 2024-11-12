import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from CLIP import CLIP
from PIL import Image

def train_clip(model):
    #add code to train the model
    pass
class Image_Text_Dataset():
    def __init__(self, image_text_pairs, image_size=224):
        self.image_text_pairs = image_text_pairs
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.image_text_pairs)
    
    def __getitem__(self, idx):
        image_path, text = self.image_text_pairs[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, text

def main():

    model = CLIP(embed_dim=128, temperature=0.1)

    #add code to make image text pairs
    image_text_pairs = [] # fill this in the format ["path/<>.jpg", "(P1:29),(P2:31)"] where P1, P2 are parameter and thei respective values

    dataset = Image_Text_Dataset(image_text_pairs)

    #add code to train model
    trained_model = train_clip(model=model)

    #save the trained model

if __name__ == "__main__":
    main()
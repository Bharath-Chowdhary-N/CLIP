import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from CLIP import CLIP
from PIL import Image
from transformers import CLIPTokenizer as HFCLIPTokenizer

def train_clip(model, device, learning_rate=1e-4, dataset):

    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data Loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    



class Image_Text_Dataset():
    def __init__(self, image_text_pairs, image_size=224, context_length=64):
        self.image_text_pairs = image_text_pairs
        self.context_length = context_length
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
        self.tokenizer = HFCLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    def __len__(self):
        return len(self.image_text_pairs)
    
    def __getitem__(self, idx):
        image_path, text = self.image_text_pairs[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        tokenized = self.tokenizer(text, padding="max_length", max_length = self.context_length, truncation=True, return_sensors="pt")
        return image, tokenized["input_ids"][0]

def main():

    model = CLIP(embed_dim=128, temperature=0.1)

    #add code to make image text pairs
    image_text_pairs = [] # fill this in the format ["path/<>.jpg", "(P1:29),(P2:31)"] where P1, P2 are parameter and thei respective values

    dataset = Image_Text_Dataset(image_text_pairs)

    #add code to train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_clip(model=model, device=device, learning_rate=1e-4, dataset=dataset)

    #save the trained model

if __name__ == "__main__":
    main()
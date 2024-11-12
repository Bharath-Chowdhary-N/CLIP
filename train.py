import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from CLIP import CLIP
from PIL import Image
from torch.nn import Functional as F
from transformers import CLIPTokenizer as HFCLIPTokenizer
import numpy as np
def train_clip(model, device, dataset, learning_rate=1e-4, batch_size=16, warmup_steps=2000, num_epochs=50, enable_amp = True):

    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data Loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    #scaler = torch.cuda.amp.grad_scaler() #use this function later

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        for batch_idx, (images, tokens) in enumerate(train_loader):
            images = images.to(device)
            tokens = tokens.to(device)
            with torch.cuda.amp.autocast():
                 logits_per_image, logits_per_text = model(images,tokens)
            
                 labels = torch.arange(len(images)).to(device)
                 loss_i = F.cross_entropy(logits_per_image, labels)
                 loss_t = F.cross_entropy(logits_per_text, labels)
                 loss = (loss_i + loss_t) / 2
            
            optimizer.zero_grad()

            loss.backward()
            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                model.logit_scale.clamp_(0,np.log(100))
            
            total_loss += loss.item()
            num_batches+=1

            if batch_idx%100==0:
                avg_loss = total_loss / num_batches
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                      f"Scale: {model.logit_scale.exp().item():.1f}"
                      )
            global_step +=1
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            #'scaler': scaler.state_dict() if enable_amp else None,
        }, f'clip_checkpoint_epoch_{epoch}.pt')
        
            print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / num_batches:.4f}")
    
    return model


            

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
    torch.save(trained_model.state_dict(), 'clip_trained_model.pt')

if __name__ == "__main__":
    main()
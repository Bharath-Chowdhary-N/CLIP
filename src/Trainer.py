import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPTokenizer as HFCLIPTokenizer
import numpy as np
import os
import tqdm

def train_clip(model, device, train_loader, learning_rate=1e-4, batch_size=16, warmup_steps=2000, num_epochs=50, enable_amp = True, dest_folder='./checkpoints'):

    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data Loader
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    def get_lr(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    #scaler = torch.cuda.amp.grad_scaler() #use this function later
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    if not dest_folder.endswith('/'):
        dest_folder = dest_folder+'/'
    
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        print(f"Epoch {epoch}")
        print("-------------------------------------------------------")

        # batch loss data
        pbar = tqdm.tqdm(train_loader, desc='Training: ', dynamic_ncols=True)
        
        for batch_idx, (images, tokens) in enumerate(pbar):
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

            
            avg_loss = total_loss / num_batches
            loss_str = "".join(f"Loss: {avg_loss:.4f}, "
                  f"Scale: {model.logit_scale.exp().item():.1f}")
            pbar.set_postfix_str(loss_str)
                
            global_step +=1
        
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / num_batches:.4f}")
        if epoch%5 ==0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                #'scaler': scaler.state_dict() if enable_amp else None,
            }, f'{dest_folder}/clip_checkpoint_epoch_{epoch}.pt')
                print(f'Saved Model {dest_folder}/clip_checkpoint_epoch_{epoch}.pt')
    return model
from src.CLIP import CLIP

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
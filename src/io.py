from PIL import Image
from transformers import CLIPTokenizer as HFCLIPTokenizer
import torch
from .functions import return_cutout, normalize_image_simulated
import numpy as np
import tqdm
from torchvision import transforms
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

class Image_Text_Dataset():
    
    def __init__(self, fits_file, cutout_size=64, image_size=224, context_length=64, do_transform=False):
        self.context_length = context_length
        self.image_size = image_size
        self.do_transform = do_transform
        self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
        self.tokenizer = HFCLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.fits_file = fits_file
        self.cutout_size=cutout_size
        image_hdu, table_hdu, _ = fits.open(fits_file, memmap=True)
        self.catalog = Table(table_hdu.data)
        self.image_data = image_hdu.data
        self.image_wcs = WCS(image_hdu.header)
        self.subset = self.catalog[np.where((self.catalog['Inferred z']>=1) & (self.catalog['Inferred z']<=3) & (np.log10(self.catalog['Stellar mass w2sr'])>=8))]
        
        contains_nans = []
        for each_galaxy in tqdm.tqdm(self.subset):
            cutout_image = return_cutout(self.image_data, each_galaxy['RA degree'], each_galaxy['DEC degree'], self.cutout_size, self.image_wcs)
            normed_image_first_pass = normalize_image_simulated(cutout_image)
            contains_nans.append((np.isnan(cutout_image).any()) | ((cutout_image == 0).all()) | (np.isnan(normed_image_first_pass).any()))
        
        self.subset_no_nans = self.subset[np.where(np.invert(np.array(contains_nans)))]
        print(f'Found {len(self.subset_no_nans)} images with no NaNs')
        
    def __len__(self):
        return(len(self.subset_no_nans))
    
    def __getitem__(self, index):
        item = return_cutout(self.image_data, self.subset_no_nans['RA degree'][index], self.subset_no_nans['DEC degree'][index], self.cutout_size, self.image_wcs)
        image_data = torch.unsqueeze(torch.Tensor(normalize_image_simulated(item).astype('float32')), dim=0)
        
        conditional_data = f"(P1:{np.round(self.subset_no_nans['Inferred z'][index],3)}), (P2:{np.round(np.log10(self.subset_no_nans['Stellar mass w2sr'][index]), 3)})"

        if self.do_transform:
            image = self.transform(image)
        
        tokenized = self.tokenizer(conditional_data, padding="max_length", max_length = self.context_length, truncation=True)
        return image_data.repeat(3,1,1), torch.Tensor(tokenized["input_ids"]).to(torch.int32) #conditional_data
    
    
        
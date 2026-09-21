import pathlib
import torch
import numpy as np
from PIL import Image

class SAnime2XDataset:
    def __init__(self, p, patch_size=96, data_augmentation=True):
        self.path = p
        self.patch_size = patch_size
        p = torch.randperm(2000)
        self.train_idx = p[:1900]
        self.val_idx = p[1900:]
        self.train_size = 1900
        self.val_size = 100
        self.data_augmentation = data_augmentation

    def __get_path_of_idx(self, i):
        p = pathlib.Path(self.path) / pathlib.Path(f'part_{(i//200)+1:02}') / pathlib.Path(f'00{i:04}.jpeg')
        return p

    def __get_image_of_idx(self, i):
        p = self.__get_path_of_idx(i)
        img = Image.open(p)
        return img

    def __get_random_crop(self, i, zoom=2):
        i = i.item()
        img = self.__get_image_of_idx(i)
        w, h = img.size
        x = torch.randint(w-self.patch_size-1, (1,)).item()
        y = torch.randint(h-self.patch_size-1, (1,)).item()
        cropped = img.crop((x, y, x+self.patch_size, y+self.patch_size))
        shrinked = cropped.resize((self.patch_size//zoom, self.patch_size//zoom),
                                  resample=Image.Resampling.BILINEAR)
        enlarged = shrinked.resize((self.patch_size, self.patch_size),
                                   resample=Image.Resampling.BILINEAR)
        cropped_tensor = torch.Tensor(np.asarray(cropped).copy()) \
                              .permute(2, 0, 1)
        enlarged_tensor = torch.Tensor(np.asarray(enlarged).copy()) \
                               .permute(2, 0, 1)
        # data augmentation
        if self.data_augmentation:
            k = torch.randint(4, (1,)).item()
            cropped_tensor = torch.rot90(cropped_tensor, k, dims=(1, 2))
            enlarged_tensor = torch.rot90(cropped_tensor, k, dims=(1, 2))
            k2 = torch.randint(4, (1,)).item()
            if k2 % 2 == 1:
                cropped_tensor = torch.flip(cropped_tensor, (1,))
                enlarged_tensor = torch.flip(enlarged_tensor, (1,))
            if k2 // 2 == 1:
                cropped_tensor = torch.flip(cropped_tensor, (2,))
                enlarged_tensor = torch.flip(enlarged_tensor, (2,))
            
        return enlarged_tensor/255, cropped_tensor/255

    def get_train_random_crop(self, n=1, zoom=2):
        # returns float [0,1]
        i_idx = torch.randint(1900, (n,))
        idx = self.train_idx[i_idx]
        enlarged = []
        cropped = []
        for i in idx:
            e, c = self.__get_random_crop(i, zoom=zoom)
            enlarged.append(e)
            cropped.append(c)
        return torch.stack(enlarged), torch.stack(cropped)

    def get_train_images(self, idx):
        # returns int [0,255]
        idx = self.train_idx[idx]
        images = []
        for i in idx:
            img = self.__get_image_of_idx(i)
            images.append(torch.Tensor(np.asarray(img).copy()) \
                          .permute(2, 0, 1))
        return images

    def get_val_images(self, idx):
        # returns int [0,255]
        idx = self.val_idx[idx]
        images = []
        for i in idx:
            img = self.__get_image_of_idx(i)
            images.append(torch.Tensor(np.asarray(img).copy()) \
                          .permute(2, 0, 1))
        return images
    
    def get_train_random_image(self, n=1):
        # returns int [0,255]
        return self.get_train_images(idx)

    def get_val_random_image(self, n=1):
        # returns int [0,255]
        i_idx = torch.randint(self.val_size, (n,))
        idx = self.val_idx[i_idx]
        images = []
        for i in idx:
            img = self.__get_image_of_idx(i)
            images.append(torch.Tensor(np.asarray(img).copy()) \
                          .permute(2, 0, 1))
        return images

    def get_val_shrink_enlarged_images(self, idx, zoom=2):
        # returns int [0,255]
        idx = self.val_idx[idx]
        images = []
        for i in idx:
            img = self.__get_image_of_idx(i)
            shrinked = img.resize((img.size[0]//zoom, img.size[1]//zoom),
                                      resample=Image.Resampling.BILINEAR)
            enlarged = shrinked.resize(img.size,
                                       resample=Image.Resampling.BILINEAR)
            images.append(torch.Tensor(np.asarray(enlarged).copy()) \
                          .permute(2, 0, 1))
        return images
        

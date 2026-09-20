import pathlib
import torch
import numpy as np
from PIL import Image

class SAnime2XDataset:
    def __init__(self, p, patch_size=96):
        self.path = p
        self.patch_size = patch_size

    def __get_random_crop(self, i):
        i = i.item()
        p = pathlib.Path(self.path) / pathlib.Path(f'part_{(i//200)+1:02}') / pathlib.Path(f'00{i:04}.jpeg')
        img = Image.open(p)
        w, h = img.size
        x = torch.randint(w-self.patch_size-1, (1,)).item()
        y = torch.randint(h-self.patch_size-1, (1,)).item()
        cropped = img.crop((x, y, x+self.patch_size, y+self.patch_size))
        shrinked = cropped.resize((self.patch_size//2, self.patch_size//2),
                                  resample=Image.Resampling.BILINEAR)
        enlarged = shrinked.resize((self.patch_size, self.patch_size),
                                   resample=Image.Resampling.BILINEAR)
        cropped_tensor = torch.Tensor(np.asarray(cropped)) \
                              .reshape((self.patch_size, self.patch_size, 3)) \
                              .permute(2, 0, 1)
        enlarged_tensor = torch.Tensor(np.asarray(enlarged)) \
                               .reshape((self.patch_size, self.patch_size, 3)) \
                               .permute(2, 0, 1)
        return enlarged_tensor/255, cropped_tensor/255

    def get_random_crop(self, n=1):
        idx = torch.randint(2000, (n,))
        enlarged = []
        cropped = []
        for i in idx:
            e, c = self.__get_random_crop(i)
            enlarged.append(e)
            cropped.append(c)
        return torch.stack(enlarged), torch.stack(cropped)
        

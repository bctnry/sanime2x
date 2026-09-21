import pathlib
import time
import torch
import numpy as np
from PIL import Image

SPLIT_SEED = 20260921  # fixed train/val split; same 100 val images every run

CACHE_FILE = 'image_cache.pt'


class SAnime2XDataset:
    # loads every image into RAM once (uint8), then serves crops by slicing.
    # ~12 GB for all 2000 images; pass cache_limit to cap how many get cached
    # (uncached images fall back to jpeg decode on the fly).
    # the cache is persisted to image_cache.pt (~12 GB on disk) so subsequent
    # runs skip the ~90s decode pass.

    def __init__(self, p, patch_size=96, cache_limit=2000, data_augmentation=True):
        self.path = p
        self.patch_size = patch_size

        gen = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(2000, generator=gen)
        self.train_idx = perm[:1900]
        self.val_idx = perm[1900:]
        self.train_size = 1900
        self.val_size = 100
        self.data_augmentation = data_augmentation

        cache = pathlib.Path(__file__).parent / CACHE_FILE
        n_avail = len(list(pathlib.Path(p).rglob('*.jpeg')))
        if cache.exists() and cache_limit >= n_avail:
            t0 = time.time()
            blob = torch.load(cache, weights_only=True)
            # only trust the cache if the dataset on disk hasn't changed
            if blob.get('source') == str(p) and len(blob['images']) == n_avail:
                self.images = blob['images']
                print(f'loaded {len(self.images)} cached images '
                      f'in {time.time()-t0:.1f}s')
                return
            print('cache stale/short -> rebuilding')
        self.images = {}
        files = sorted(pathlib.Path(p).rglob('*.jpeg'))
        assert len(files) == 2000, f'expected 2000 images, found {len(files)}'
        t0 = time.time()
        for k, f in enumerate(files):
            if k >= cache_limit:
                break
            img = np.asarray(Image.open(f)).copy()
            # store as (C, H, W) uint8 for fast slicing; lazy conversion to float
            self.images[k] = torch.from_numpy(img).permute(2, 0, 1)
        print(f'cached {len(self.images)} images in {time.time()-t0:.1f}s')
        if len(self.images) == n_avail:
            torch.save({'source': str(p), 'images': self.images}, cache)

    def __get_image(self, i):
        if i in self.images:
            return self.images[i]
        img = Image.open(self.__path_of(i))
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)

    def __path_of(self, i):
        return pathlib.Path(self.path) / f'part_{(i//200)+1:02}' / f'00{i:04}.jpeg'

    def __crop_pair(self, img, ps, zoom=2):
        _, h, w = img.shape
        x = torch.randint(w - ps - 1, (1,)).item()
        y = torch.randint(h - ps - 1, (1,)).item()
        cropped = img[:, y:y+ps, x:x+ps]
        c_img = Image.fromarray(cropped.permute(1, 2, 0).numpy())
        shrinked = c_img.resize((ps//zoom, ps//zoom), resample=Image.Resampling.BILINEAR)
        enlarged = shrinked.resize((ps, ps), resample=Image.Resampling.BILINEAR)
        cropped_t = cropped.to(torch.float).div_(255)
        enlarged_t = torch.from_numpy(np.asarray(enlarged).copy()) \
                          .permute(2, 0, 1).to(torch.float).div_(255)
        
        # data augmentation
        if self.data_augmentation:
            k = torch.randint(4, (1,)).item()
            cropped_t = torch.rot90(cropped_t, k, dims=(1, 2))
            enlarged_t = torch.rot90(enlarged_t, k, dims=(1, 2))
            k2 = torch.randint(4, (1,)).item()
            if k2 % 2 == 1:
                cropped_t = torch.flip(cropped_t, (1,))
                enlarged_t = torch.flip(enlarged_t, (1,))
            if k2 // 2 == 1:
                cropped_t = torch.flip(cropped_t, (2,))
                enlarged_t = torch.flip(enlarged_t, (2,))
                
        return enlarged_t, cropped_t

    def get_train_random_crop(self, n=1, zoom=2):
        # return float [0,1]
        ii = self.train_idx[torch.randint(self.train_size, (n,))]
        es, cs = [], []
        for i in ii:
            img = self.__get_image(int(i))
            e, c = self.__crop_pair(img, self.patch_size, zoom=zoom)
            es.append(e)
            cs.append(c)
        return torch.stack(es), torch.stack(cs)

    def get_val_images(self, idx):
        # returns int [0,255]
        return [self.__get_image(int(self.val_idx[k])) for k in idx]

    def get_val_shrink_enlarged_images(self, idx, zoom=2):
        # returns int [0,255]
        out = []
        for k in idx:
            img = self.__get_image(int(self.val_idx[k]))
            pil = Image.fromarray(
                img.permute(1, 2, 0).numpy())
            shrinked = pil.resize((pil.size[0]//zoom, pil.size[1]//zoom),
                                  resample=Image.Resampling.BILINEAR)
            enlarged = shrinked.resize(pil.size,
                                       resample=Image.Resampling.BILINEAR)
            out.append(torch.from_numpy(np.asarray(enlarged).copy())
                       .permute(2, 0, 1))
        return out

    def get_val_random_image(self, n=1):
        return self.get_val_images(torch.randint(self.val_size, (n,)))

    def get_train_images(self, idx):
        # returns int [0, 255]
        return [self.__get_image(int(self.train_idx[k])) for k in idx]

    def get_train_random_image(self, n=1):
        # returns int [0, 255]
        return self.get_train_images(torch.randint(self.train_size, (n,)))

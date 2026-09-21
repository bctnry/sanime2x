import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model_resnet import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patch_size = 96
n_block = 8
margin = 2 + 2 * n_block + 2

model = ResNet(channel=64, n_block=8)
model.load_state_dict(torch.load('model_resnet.pt', map_location='cpu', weights_only=True)['model'])
model = model.to(device)
model.eval()

miku = Image.open('miku2.png')
# miku = Image.open('img2.jpg')
miku_enlarged = miku.resize((miku.size[0]*2, miku.size[1]*2),
                            resample=Image.Resampling.BILINEAR)
miku_enlarged_bicubic = miku.resize((miku.size[0]*2, miku.size[1]*2),
                            resample=Image.Resampling.BICUBIC)
miku_enlarged_bicubic.save('miku_enlarged.jpeg')

img = torch.from_numpy(np.asarray(miku_enlarged).copy()) \
           .permute(2, 0, 1).to(torch.float).div_(255)      # (3, H, W)
H, W = img.shape[1], img.shape[2]

# reflect-pad once so every tile can get `margin` of real context at borders
padded = F.pad(img.unsqueeze(0), (margin,)*4, mode='reflect')[0]
import torchvision
torchvision.utils.save_image(padded, 'padded.png')

out = torch.zeros_like(img)
with torch.no_grad():
    for r0 in range(0, H, patch_size):
        for c0 in range(0, W, patch_size):
            r1, c1 = min(r0 + patch_size, H), min(c0 + patch_size, W)
            window = padded[:, r0:r1 + 2*margin, c0:c1 + 2*margin] \
                          .unsqueeze(0).to(device)
            result = model(window)
            out[:, r0:r1, c0:c1] = result[0,
                :, margin:margin + (r1 - r0), margin:margin + (c1 - c0)].cpu()

import torchvision
torchvision.utils.save_image(out.clamp(0, 1), 'sharpened.png')

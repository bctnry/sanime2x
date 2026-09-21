import torch
import torch.nn as nn
import torch.nn.functional as F
from model_resnet import ResNet
from dataloader2 import SAnime2XDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_block = 8
channel = 64
lr_peak = 2.5e-4
n_steps = 100000

batch_size = 8
patch_size = 96

ds = SAnime2XDataset('/home/bctnry/s/sanime2x-2k/')

model = ResNet(channel=channel, n_block=n_block)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_peak)
# math.log(0.1) / math.log(0.9) = ~21.85
#     --> about ~21.85 steps of multiplying by 0.9 to goes from full to 0.1
# thus 100000 // 21.85 = 4575
import math
step_width = n_steps // (math.log(0.1) / math.log(0.9))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_width, gamma=0.9)

PSNR_EVAL_IDX = torch.randint(ds.val_size, (2,))
def val_psnr(model):
    calc = []
    idx = PSNR_EVAL_IDX
    images = ds.get_val_images(idx)
    se_images = ds.get_val_shrink_enlarged_images(idx)
    model.eval()
    with torch.no_grad():
        for k, img in enumerate(images):
            se_img = se_images[k]
            h, w = img.shape[1], img.shape[2]
            p_h, p_w = h // patch_size, w // patch_size
            # we can largely ignore the edges without compromising
            # too much about the accuracy of this measuring.
            for j in range(0, p_h):
                for i in range(0, p_w):
                    cropped = img[:,
                                  j*patch_size:(j+1)*patch_size,
                                  i*patch_size:(i+1)*patch_size]
                    cropped = cropped.reshape(1, *cropped.shape)
                    cropped = cropped.to(device)
                    se_cropped = se_img[:,
                                        j*patch_size:(j+1)*patch_size,
                                        i*patch_size:(i+1)*patch_size]
                    se_cropped = se_cropped.reshape(1, *se_cropped.shape)
                    se_cropped = se_cropped.to(device) / 255
                    res = model(se_cropped)
                    cropped = cropped / 255
                    
                    loss = F.mse_loss(res, cropped)
                    calc.append(loss.item())
    total_loss = sum(calc) / len(calc)
    psnr = -10*torch.log10(torch.Tensor([total_loss])).item()
    return psnr

losses = []
avg_span = 50
last_psnr = 0
for i in range(n_steps):
    model.train()
    
    enlarged, cropped = ds.get_train_random_crop(n=batch_size)
    enlarged = enlarged.to(device)
    cropped = cropped.to(device)
    calculated = model(enlarged)
        
    loss = F.l1_loss(calculated, cropped)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
        
    losses.append(loss.item())
    if len(losses) > avg_span:
        losses = losses[1:]

    avgloss = sum(losses)/len(losses)
    print(f'Step {i+1}: loss={loss.item():.4f} avgloss={avgloss:.4f} last_psnr={last_psnr:.4f} dB')
    # print(f'Step {i+1}: loss={loss.item():.4f} avgloss={avgloss:.4f}')
    if i % 500 == 499:
    # if i % 100 == 99:
        last_psnr = val_psnr(model)
    if i % 100 == 99:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, 'model_resnet.pt')
                   
torch.save({
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
}, 'model_resnet.pt')
    


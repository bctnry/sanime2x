import torch
import torch.nn as nn
import torch.nn.functional as F
from model_cnn import PlainCNN
from dataloader import SAnime2XDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_layer = 12
lr_peak = 2e-4
n_steps = 10000

batch_size = 8

ds = SAnime2XDataset('/home/bctnry/s/sanime2x-2k/')

model = PlainCNN(n_layer)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_peak)

losses = []
avg_span = 50
for i in range(n_steps):
    enlarged, cropped = ds.get_random_crop(n=batch_size)
    enlarged = enlarged.to(device)
    cropped = cropped.to(device)
    calculated = model(enlarged)
        
    loss = F.mse_loss(calculated, cropped)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    losses.append(loss.item())
    if len(losses) > avg_span:
        losses = losses[1:]

    avgloss = sum(losses)/len(losses)
    psnr = -10*torch.log10(torch.Tensor([avgloss])).item()
    print(f'Step {i+1}: loss={loss.item():.4f} avgloss={avgloss:.4f} psnr={psnr:.4f} dB')
    if i % 100 == 99:
        torch.save({
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, 'model_cnn.pt')
                   
torch.save({
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
}, 'model_cnn.pt')
    


import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
from tqdm import tqdm

import utils
import unet
from dataloader import load_train_data
import diffusion_solver


# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "OCTA_data.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = ["octa500"]
num_sample = 100
batch_size = 24
p_size = [256, 256]
intensity_range = [-1, 1]

train_data = load_train_data(data, p_size, num_sample, datasets, intensity_range, batch_size=batch_size)


# diffusion configuration
beta_start = 0.0001
beta_end = 0.02
T = 300
betas = diffusion_solver.get_beta_schedule(beta_schedule="linear",
                                           beta_start=beta_start,
                                           beta_end=beta_end,
                                           num_diffusion_timesteps=T)

sampler = diffusion_solver.DiffusionSampler(betas, device=device)



# ------------------------------ load model ------------------------------
model = unet.SimpleUnet().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))

# training configuration
n_epoch = 300
mse_loss = nn.MSELoss()

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


for epoch in range(n_epoch):

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):
            optimizer.zero_grad()

            # forward sample
            x_0 = x[:,0,:,:].unsqueeze(1)
            t = torch.randint(0, T, (x_0.shape[0],)).long()
            x_t, eps = sampler.forward_sample(x_0, t)

            # model prediction
            eps_pred = model(x_t.to(device), t.to(device))
            
            loss = F.l1_loss(eps.to(device), eps_pred)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, L1_loss: %.4f" %(epoch, loss.item()))

            if step % 1000 == 0:
                x_t = torch.randn((1, 1, 256, 256))
                x_0 = sampler.reverse_iterate(x_t, T-1, model)
                
                img = utils.tensor2pil(x_0)
                save_name = f"epoch_{epoch}_step_{step}"
                utils.image_saver(np.array(img), save_path, save_name)

        scheduler.step()
    
    name = "unconditional_diffusion_octa.pt"
    torch.save(model.state_dict(), ckpt_path + name)





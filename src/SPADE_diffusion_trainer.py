import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import utils
import diffusion_solver 
from dataloader import load_train_data


device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

#%% load data and diffusion sampler

with open(data_path + "FP_data.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = ["drive", "stare", "chase",
            "hrf_control", "hrf_diabetic", "hrf_glaucoma",
            "aria_control", "aria_diabetic", "aria_amd",]
num_sample = 80
batch_size = 16
p_size = [256, 256]
intensity_range = [-1, 1]

train_data = load_train_data(data, p_size, num_sample, datasets, intensity_range, batch_size=batch_size)


# diffusion configuration
beta_start = 0.0001
beta_end = 0.05
num_timesteps = 100
betas = diffusion_solver.get_beta_schedule(beta_schedule="sigmoid",
                                            beta_start=beta_start,
                                            beta_end=beta_end,
                                            num_diffusion_timesteps=num_timesteps).to(device)

sampler = diffusion_solver.DiffusionSampler(betas, device=device)

#%% load model
encoder_nchs = [4, 8, 16, 32, 64]
t_embed_dim = 32

model = diffusion_solver.DiffusionModel(input_nch=3, 
                                        encoder_nchs=encoder_nchs, 
                                        p_size=p_size, 
                                        t_embed_dim=t_embed_dim).to(device)

#%% training configuration
n_epoch = 300
mse_loss = nn.MSELoss()

lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5,0.999))
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)


for epoch in range(n_epoch):

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):
            # randomly select t
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            t = torch.randint(0, num_timesteps, (batch_size, ), device=device, dtype=torch.float32)#.long()

            # forward process
            x_t, eps = sampler.forward_sample(x, t)
            
            # reverse process
            pred = model(x_t, y, t)

            # loss
            loss = mse_loss(pred, eps)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, MSE_loss: %.4f" %(epoch, loss.item()))

        scheduler.step()

        name = "SPADE_diffusion.pt"
        torch.save(model.state_dict(), ckpt_path + name)


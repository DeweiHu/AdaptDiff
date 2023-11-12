import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from dataloader import load_train_data
import diffusion_solver
import unet


data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "FP_data.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = ["drive", "stare", "chase",
            "hrf_control", "hrf_diabetic", "hrf_glaucoma",
            "aria_control", "aria_diabetic", "aria_amd",]
num_sample = 100
batch_size = 8
p_size = [256, 256]
intensity_range = [-1, 1]

train_data = load_train_data(data, p_size, num_sample, datasets, intensity_range, batch_size=batch_size)


# ---------------------- training ----------------------
device = torch.device("cuda")
model = unet.SimpleUnet().to(device)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 300

T = 100
beta_start = 0.0001
beta_end = 0.04
betas = diffusion_solver.get_beta_schedule(beta_schedule="sigmoid",
                                            beta_start=beta_start,
                                            beta_end=beta_end,
                                            num_diffusion_timesteps=T)

sampler = diffusion_solver.DiffusionSampler(betas, device=device)


for epoch in range(epochs):

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:
    
        for step, (x, y) in enumerate(train_data):
            optimizer.zero_grad()

            t = torch.randint(0, T, (batch_size, )).long()
            x_t, eps = sampler.forward_sample(x, t)

            eps_pred = model(x_t.to(device), t.to(device))

            loss = mse_loss(eps_pred, eps.to(device))
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, MSE_loss: %.4f" %(epoch, loss.item()))

        scheduler.step()

        # evaluation
        seed = torch.randn((1, 3, 256, 256), dtype=torch.float32)
        file_name = f"epoch_{epoch}"
        sampler.grid_plot(seed, model, save_path+file_name)


    name = "diffusion_unet_FP.pt"
    torch.save(model.state_dict(), ckpt_path + name)
    

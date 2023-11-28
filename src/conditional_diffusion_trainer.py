import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
import random
from tqdm import tqdm

import utils
from dataloader import load_train_data
import diffusion_solver


# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "OCTA_data.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = ["octa500", "rose"]
# datasets = ["drive", "stare", "chase",
#             "hrf_control", "hrf_diabetic", "hrf_glaucoma",
#             "aria_control", "aria_diabetic", "aria_amd",]

num_sample = 80
batch_size = 8
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

sampler = diffusion_solver.DiffusionSampler(betas, 
                                            device=device, 
                                            mode='conditional')



# ------------------------------ load model ------------------------------
# encoder_nchs = [8, 16, 32, 64, 128]
# t_embed_dim = 32

# model = diffusion_solver.DiffusionModel(input_nch=1, 
#                                         encoder_nchs=encoder_nchs, 
#                                         p_size=p_size, 
#                                         t_embed_dim=t_embed_dim).to(device)

model = diffusion_solver.condition_Unet().to(device)

# training configuration
n_epoch = 200
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
            x_0 = x[:,1,:,:].unsqueeze(1)
            t = torch.randint(0, T, (x_0.shape[0],)).long()
            x_t, eps = sampler.forward_sample(x_0, t)

            # classifier-free guidance
            seed = random.uniform(0, 1)
            if seed >= 0.7:
                y = torch.zeros_like(y)

            # model prediction
            eps_pred = model(x_t.to(device), y.to(device), t.to(device))
            
            loss = F.l1_loss(eps.to(device), eps_pred)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, L1_loss: %.4f" %(epoch, loss.item()))

        # plot results
        x_t = torch.randn((1, 1, 256, 256))
        x_condition = y[0].unsqueeze(0)
        img = sampler.get_conditonal_diffusion_result(x_t, x_condition, model)
        
        save_name = f"epoch_{epoch}"
        utils.image_saver(img, save_path, save_name)

        scheduler.step()
    
    name = "conditional_diffusion_octa_(clf).pt"
    torch.save(model.state_dict(), ckpt_path + name)





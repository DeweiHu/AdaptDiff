import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
from tqdm import tqdm
import random

import utils
from dataloader import load_train_data
import diffusion_solver
import models

# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "OCTA_data.pickle", "rb") as handle:
    data = pickle.load(handle)

print(list(data))

datasets = ["octa500_6M"]

num_sample = 50
batch_size = 4
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
seg_model = models.res_UNet([8, 32, 32, 64, 64, 16], 1, 2).to(device)
seg_model.load_state_dict(torch.load(ckpt_path + "rUNet.pt"))

model = diffusion_solver.condition_Unet().to(device)

# training configuration
n_epoch = 20
mse_loss = nn.MSELoss()

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

softmax = nn.Softmax2d()

for epoch in range(n_epoch):

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):

            optimizer.zero_grad()

            # forward sample
            x_0 = x[:,1,:,:].unsqueeze(1)
            t = torch.randint(0, T, (x_0.shape[0],)).long()
            x_t, eps = sampler.forward_sample(x_0, t)

            pred_y, _ = seg_model(x_0.to(device))
            pred_y = torch.argmax(softmax(pred_y), dim=1).unsqueeze(1).to(torch.float32)

            # model prediction
            eps_pred = model(x_t.to(device), pred_y.to(device), t.to(device))
            
            loss = F.l1_loss(eps.to(device), eps_pred)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, L1_loss: %.4f" %(epoch, loss.item()))

        # plot results
        x_t = torch.randn((1, 1, p_size[0], p_size[1]))
        x_condition = y[0].unsqueeze(0)
        img, label = sampler.get_conditonal_diffusion_result(x_t, x_condition, model)
        
        img = utils.ImageRescale(img, [0, 1])
        template = utils.ImageRescale(x[0, 0, :, :].numpy(), [0, 1])
        img_hm = utils.hist_match(img, template)

        img_hm = np.uint8(utils.ImageRescale(img_hm, [0, 255]))
        label = np.uint8(utils.ImageRescale(label, [0, 255]))

        save_name = f"epoch_{epoch}"
        utils.image_saver(np.hstack((img_hm, label)), save_path, save_name)

        scheduler.step()
    
    name = f"diffusion({datasets[0]}).pt"
    torch.save(model.state_dict(), ckpt_path + name)





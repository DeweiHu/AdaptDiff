import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
import random
from tqdm import tqdm

import utils
import dataloader
import losses

import sys
sys.path.insert(0, "/media/dewei/New Volume/static representation/src/")
import models

# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

dataset = "octa500"
batch_size = 5

with open(data_path + f"{dataset}_syn.pickle", "rb") as handle:
    data = pickle.load(handle)

train_data = dataloader.load_synthetic_data(data, batch_size)


# ------------------------------ load model ------------------------------
model_root = "/media/dewei/New Volume/Model/"
seg_model = models.res_UNet([8,32,32,64,64,16], 1, 2).to(device)
seg_model.load_state_dict(torch.load(model_root + "rUNet.pt"))

# training configuration
n_epoch = 30
DSC_loss = losses.DiceBCELoss()
CE_loss = nn.CrossEntropyLoss()

lr = 1e-3
optimizer = torch.optim.Adam(seg_model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

softmax = nn.Softmax2d()

# ------------------------------ train model ------------------------------

for epoch in range(n_epoch):

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):
            
            seg_model.train()
            optimizer.zero_grad()

            x = x.to(device)
            y = y.squeeze(1).to(device)

            pred, _ = seg_model(x)
            pred_y = torch.argmax(softmax(pred), dim=1)
            
            ce_loss = CE_loss(pred, y)
            dsc_loss = DSC_loss(pred_y, y)
            seg_loss = ce_loss + dsc_loss
            seg_loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("Epoch: %d. ce-loss: %.4f. dsc-loss: %.4f." \
                                 %(epoch+1, ce_loss.item(), dsc_loss.item()))

        scheduler.step()

    name = f"finetune_{dataset}.pt"
    torch.save(seg_model.state_dict(), ckpt_path + name)


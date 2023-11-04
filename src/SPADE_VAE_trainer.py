import utils
import modules
import losses
from dataloader import load_train_data

import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/result_vae_new/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

#%% Model configuration
encoder_nchs = [4, 8, 16, 32, 64]
p_size = [256, 256]
vae = modules.Semantic_Generator(input_nch=3, 
                                 encoder_nchs=encoder_nchs,
                                 H=p_size[0],
                                 W=p_size[1]).to(device)

# vae.load_state_dict(torch.load(ckpt_path + "SPADE_generator.pt"))

#%% Data configuration

with open(data_path + "FP_data.pickle", "rb") as handle:
    data = pickle.load(handle)

num_sample = 80
batch_size = 16
datasets = ["drive", "stare", "chase",
            "hrf_control", "hrf_diabetic", "hrf_glaucoma",
            "aria_control", "aria_diabetic", "aria_amd",]

#%% Training 
num_epoch = 200
real_label = 1.
fake_label = 0.
kld_loss = losses.KLDLoss()
l1_loss = nn.L1Loss()

optimizer = torch.optim.AdamW(vae.parameters(), lr=5e-4)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)


for epoch in range(num_epoch):

    # re-sample patches for each epoch
    train_data = load_train_data(data, p_size, num_sample, datasets, batch_size=batch_size)

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            fake_im, mu, logvar= vae(x, y)

            optimizer.zero_grad()

            loss = kld_loss(mu, logvar) + 100 * l1_loss(fake_im, x)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, KLD_loss: %.4f, L1_loss: %.4f" 
                                 %(epoch, kld_loss(mu, logvar).item(), l1_loss(fake_im, x).item()))

        scheduler.step()
    
        with torch.no_grad():
            idx = 0
            img = utils._to_image_(fake_im[idx])
            gt = utils._to_image_(y[idx, 0, :, :])

            utils.image_saver(img, save_path, f"synthesis_{epoch}")
            utils.image_saver(gt, save_path, f"semantic_{epoch}")

        name = f'SPADE_VAE.pt'
        torch.save(vae.state_dict(), ckpt_path+name)

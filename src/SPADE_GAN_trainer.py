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
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

#%% Model configuration
encoder_nchs = [4, 8, 16, 32, 64]
p_size = [256, 256]
generator = modules.Semantic_Generator(input_nch=3, 
                                       encoder_nchs=encoder_nchs,
                                       H=p_size[0],
                                       W=p_size[1]).to(device)
discriminator = modules.Discriminator(4).to(device)

# generator.load_state_dict(torch.load(ckpt_path + "SPADE_generator.pt"))
discriminator.load_state_dict(torch.load(ckpt_path + "SPADE_discriminator.pt"))

#%% Data configuration

with open(data_path + "FP_data.pickle", "rb") as handle:
    data = pickle.load(handle)

num_sample = 80
batch_size = 16
datasets = ["drive", "stare", "chase", "prime_fp",
            "hrf_control", "hrf_diabetic", "hrf_glaucoma",
            "aria_control", "aria_diabetic", "aria_amd",]

#%% Training 
num_epoch = 300
real_label = 1.
fake_label = 0.
bce_loss = nn.BCEWithLogitsLoss()
kld_loss = losses.KLDLoss()

optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=5e-3)
scheduler_generator = StepLR(optimizer_generator, step_size=2, gamma=0.5)

optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
scheduler_discriminator = StepLR(optimizer_discriminator, step_size=2, gamma=0.7)


for epoch in range(num_epoch):

    # re-sample patches for each epoch
    train_data = load_train_data(data, p_size, num_sample, datasets, batch_size=batch_size)

    values = range(len(train_data))
    with tqdm(total=len(values)) as pbar:

        for step, (x, y) in enumerate(train_data):
            x = Variable(x).to(device)
            y = Variable(y).to(device)

            # generate fake images
            fake_im, mu, logvar= generator(x, y)

            fake_concat = torch.cat([fake_im, y], dim=1)
            real_concat = torch.cat([x, y], dim=1)
            # fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

            # ----------------- Update Discriminator -----------------
            optimizer_discriminator.zero_grad()

            # real pair
            pred_real = discriminator(real_concat).view(-1)
            gt_real = torch.full((pred_real.size(0),), real_label, device=device, dtype=torch.float32)
            loss_real = bce_loss(pred_real, gt_real)

            # fake pair
            pred_fake = discriminator(fake_concat.detach()).view(-1)
            gt_fake = torch.full((pred_fake.size(0),), fake_label, device=device, dtype=torch.float32)
            loss_fake = bce_loss(pred_fake, gt_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_discriminator.step()


            # ----------------- Update Generator -----------------
            optimizer_generator.zero_grad()

            pred_fake_for_G = discriminator(fake_concat).view(-1)
            gt_fake_for_G = torch.full((pred_fake_for_G.size(0),), real_label, device=device, dtype=torch.float32)
            loss_G = bce_loss(pred_fake_for_G, gt_fake_for_G) + kld_loss(mu, logvar)

            loss_G.backward()
            optimizer_generator.step()
            
            pbar.update(1)
            pbar.set_description("epoch: %d, G_loss: %.4f, D_loss: %.4f" %(epoch, loss_G.item(), loss_D.item()))


        scheduler_discriminator.step()
        scheduler_generator.step()
    
        with torch.no_grad():
            idx = 0
            img = utils._to_image_(fake_im[idx])
            gt = utils._to_image_(y[idx, 0, :, :])

            utils.image_saver(img, save_path, f"synthesis_{epoch}")
            utils.image_saver(gt, save_path, f"semantic_{epoch}")

        name_G = f'SPADE_generator.pt'
        torch.save(generator.state_dict(), ckpt_path+name_G)
        name_D = f'SPADE_discriminator.pt'
        torch.save(discriminator.state_dict(), ckpt_path+name_D)

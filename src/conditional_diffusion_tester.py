import torch
from torch import nn
import pickle
import numpy as np
from tqdm import tqdm

import utils
from BinaryMaskSampler import load_binary_masks
import diffusion_solver



# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "FP_data.pickle", "rb") as handle:
    data = pickle.load(handle)

mask_loader = load_binary_masks(data=data, 
                                num_sample=1, 
                                batch_size=1)

del data

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
model = diffusion_solver.condition_Unet().to(device)
model.load_state_dict(torch.load(ckpt_path + "semicond_diffusion(octa500).pt"))

# ------------------------------ test model ------------------------------

values = range(len(mask_loader))
with tqdm(total=len(values)) as pbar:

    for step, mask in enumerate(mask_loader):

        x_t = torch.randn((1, 1, 256, 256))
        mask = mask[0].unsqueeze(0)
        
        x_0 = sampler.reverse_iterate(x_t, T-1, model, mask)
        img = np.array(utils.tensor2pil(x_0.detach().cpu()))
        img = img[:, :, 0]

        mask = utils.ImageRescale(mask.detach().numpy(), [0, 255])
        mask = np.uint8(mask[0, 0, :, :])

        save_name = f"{step}"
        utils.image_saver(img, save_path + "syn_octa500", save_name)
        utils.image_saver(mask, save_path + "binary_mask", save_name)

        pbar.update(1)
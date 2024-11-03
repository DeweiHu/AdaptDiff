import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.utils as utils
import src.dataloader as dataloader
import src.diffusion_solver as diffusion_solver
import src.models as models

# ------------------------------ load data ------------------------------
device = torch.device("cuda")

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/result/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"


with open(data_path + "OCTA_data.pickle", "rb") as handle:
    target_data = pickle.load(handle)

mask_loader = dataloader.load_binary_masks(data=target_data, 
                                           num_sample=50, 
                                           batch_size=1)

target = "rose"

# template = dataloader.GetHistogramTemplate(im_list=target_data[target+"_im"],
#                                            num_sample=20,
#                                            intensity_range=[0, 1])

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
model = models.condition_Unet().to(device)
model.load_state_dict(torch.load(ckpt_path + f"{target}.pt"))

# ------------------------------ test model ------------------------------

synthesized_pair = []

values = range(len(mask_loader))
with tqdm(total=len(values)) as pbar:

    for step, mask in enumerate(mask_loader):

        # get a histogram template
        # idx = random.randint(0, template.len-1)
        # im_template = template.im[idx]

        x_t = torch.randn((1, 1, 384, 384))
        x_0 = sampler.reverse_iterate(x_t, T-1, model, mask)

        mask = utils.tensor2numpy(mask[0].detach().cpu())

        im = utils.tensor2numpy(x_0[0].detach().cpu())
        # im_hm = utils.hist_match(im, im_template[0])

        synthesized_pair.append((im, mask))
        # synthesized_pair.append((im_hm, mask))

        if step % 50 == 0:
            
            im = utils.ImageRescale(im, [0, 255])
            # im_hm = utils.ImageRescale(im_hm, [0, 255])
            mask = utils.ImageRescale(mask, [0, 255])

            utils.image_saver(img=np.hstack((im, mask)), 
                              path=save_path, 
                              name=f"{target}_{step // 100}")
        
        pbar.update(1)

with open(data_path + f"{target}_FPmask.pickle", "wb") as handle:
    pickle.dump(synthesized_pair, handle)
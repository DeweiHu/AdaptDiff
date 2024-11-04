import argparse
import torch
import os
import numpy as np

import src.utils as utils
import src.diffusion_solver as diffusion_solver
import src.models as models



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get directories")
    
    parser.add_argument('--mask_path', type=str, help="mask directory")
    parser.add_argument('--mask_name', type=str, help="binary mask name")

    parser.add_argument('--ckpt_path', type=str, help="model checkpoint directory")
    parser.add_argument('--ckpt_name', type=str, help="checkpoint name")

    parser.add_argument('--save_path', type=str, help="save directory")
    parser.add_argument('--save_name', type=str, help="image save name")    

    args = parser.parse_args()

    # ------------------------ load diffusion config -------------------------
    device = torch.device("cuda")

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
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.ckpt_name)))


    # model inference
    mask = utils.load_binary_mask(args.mask_path, args.mask_name)
    mask = torch.tensor(mask[None, None, :, :]).type(torch.FloatTensor)

    x_t = torch.randn((1, 1, 384, 384))
    x_0 = sampler.reverse_iterate(x_t, T-1, model, mask)

    im = utils.tensor2numpy(x_0[0].detach().cpu())
    im = utils.ImageRescale(im, [0, 255])

    # save the synthesized image
    utils.image_saver(img=im,
                      path=args.save_path,
                      name=args.save_name)
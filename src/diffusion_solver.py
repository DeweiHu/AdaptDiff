import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )

    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )

    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    else:
        raise NotImplementedError(beta_schedule)
    
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.tensor(betas)


class DiffusionSampler:

    def __init__(self, betas):
        super(DiffusionSampler, self).__init__()

        self.device = "cuda"
        
        # given the beta schedule, compute the close-form expressions 
        self.alphas = 1. - betas
        self.sqrt_reciprocal_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # pad 1 to the left and keep the dimension unchanged
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

    def forward_sample(self, x_0, t):
        eps = torch.randn_like(x_0).to(self.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(vals=self.sqrt_alphas_cumprod, 
                                                         t=t, 
                                                         x_shape=x_0.shape).to(self.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(vals=self.sqrt_one_minus_alphas_cumprod,
                                                                   t=t,
                                                                   x_shape=x_0.shape).to(self.device)
        x_t = sqrt_alphas_cumprod_t * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t * eps
        return x_t, eps

    
    def get_index_from_list(self, vals, t, x_shape):
        '''
          vals: [num_timestep]
             t: [batch_size]
             x: [batch_size, channel, H, W]
        output: [batch_size, 1, 1, 1]
        '''
        batch_size = t.shape[0]
        output = vals.gather(-1, t.cpu())
        output = output.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return output



if __name__ == "__main__":

    save_path = "/home/dewei/Medical_Semantic_Diffusion/"
    
    # configuration
    beta_start = 0.0001
    beta_end = 0.02
    num_timesteps = 100
    
    betas = get_beta_schedule(beta_schedule="sigmoid",
                              beta_start=beta_start,
                              beta_end=beta_end,
                              num_diffusion_timesteps=num_timesteps)
    
    # x = np.linspace(0, num_timesteps-1, num_timesteps)

    # plt.scatter(x, betas)
    # plt.savefig(save_path+"beta.png", format='png',dpi=100)

    vals = torch.linspace(0, 9, 10)
    vals_pad = F.pad(vals[:-1], (1, 0), value=1.0)

    print(f"{vals.shape}, {vals}")
    print(f"{vals_pad.shape}, {vals_pad}")


    




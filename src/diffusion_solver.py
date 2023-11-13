import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import modules

import utils


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float32,
            )
            ** 2
        )

    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float32
        )

    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float32
        )

    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps, dtype=np.float32)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    else:
        raise NotImplementedError(beta_schedule)
    
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.tensor(betas)


class DiffusionSampler:

    def __init__(self, betas, device):
        super(DiffusionSampler, self).__init__()

        self.device = device
        
        # given the beta schedule, compute the close-form expressions 
        self.betas = betas
        self.alphas = 1. - betas
        self.sqrt_reciprocal_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # pad 1 to the left and keep the dimension unchanged
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

    # set the forward sample done on cpu by default
    def forward_sample(self, x_0, t):
        eps = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(vals=self.sqrt_alphas_cumprod, 
                                                         t=t, 
                                                         x_shape=x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(vals=self.sqrt_one_minus_alphas_cumprod,
                                                                   t=t,
                                                                   x_shape=x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * eps
        return x_t, eps


    @torch.no_grad()
    def reverse_sample(self, x_t, t, model):
        
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        sqrt_reciprocal_alphas_t = self.get_index_from_list(
            self.sqrt_reciprocal_alphas, t, x_t.shape
        )
        
        posterior_variance_t = self.get_index_from_list(
            self.posterior_variance, t, x_t.shape
        )

        eps_pred = model(x_t.to(self.device), t.to(self.device)).cpu()
        
        mean = sqrt_reciprocal_alphas_t * (x_t - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t)
        noise = torch.randn_like(x_t)

        return mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def reverse_iterate(self, x_t, t, model):
        # iterative reverse sampling
        for i in range(0, t+1):
            t_tensor = torch.full((1, ), t-i, dtype=torch.long)
            x_t = self.reverse_sample(x_t, t_tensor, model)
            x_t = torch.clamp(x_t, -1.0, 1.0)

        return x_t
    

    @torch.no_grad()
    def grid_plot(self, x_t, model, save_path):
        num_plot = 10
        T = len(self.betas)
        stepsize = int(T / num_plot)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for t in range(0, T, stepsize):
            x_0 =self.reverse_iterate(x_t, t, model)

            r = int(t / stepsize) // 5
            c = int(t / stepsize) % 5
            ax = axes[r, c]
            ax.imshow(utils.tensor2pil(x_0.detach().cpu()))
            ax.set_title(f"denoise step = {t}")
            ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)


    def get_index_from_list(self, vals, t, x_shape):
        '''
          vals: [num_timestep]
             t: [batch_size]
             x: [batch_size, channel, H, W]
        output: [batch_size, 1, 1, 1]
        '''
        batch_size = t.shape[0]
        output = vals.gather(-1, t.type(torch.int64).cpu())
        output = output.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return output


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
    

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, 
                                            dtype=torch.float32, 
                                            device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionEncoder(nn.Module):

    def __init__(self, input_nch, hidden_nchs, H, W, t_embed_dim):
        super(DiffusionEncoder, self).__init__()

        self.input_nch = input_nch
        self.hidden_nchs = hidden_nchs

        # get the feature size
        c, h, w = self.get_feature_size(H, W)
        feature_nch = c * h * w
        
        self.linear = nn.Linear(feature_nch, 256)

        self.activate = nn.GELU()

        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.GELU(),
        )

        # module list
        self.Conv = nn.ModuleList()
        self.Time_mlp = nn.ModuleList()
        self.Dsample = nn.ModuleList()

        for i in range(len(self.hidden_nchs)):
            if i  == 0:
                self.Conv.append(modules.ConvBlock(input_nch, hidden_nchs[i]))
            else:
                self.Conv.append(modules.ConvBlock(hidden_nchs[i-1], hidden_nchs[i]))

            self.Time_mlp.append(nn.Linear(t_embed_dim, hidden_nchs[i]))
            self.Dsample.append(modules.DownSample(hidden_nchs[i], hidden_nchs[i]))


    def get_feature_size(self, H, W):
        c = self.hidden_nchs[-1]
        h = H // (2 ** (len(self.hidden_nchs)))
        w = W // (2 ** (len(self.hidden_nchs)))
        return c, h, w
        

    def forward(self, x, t):
        # sinosoidal embedding
        t = self.t_embedding(t)
        
        for i in range(len(self.hidden_nchs)):
            layer_output = self.Conv[i](x)
            x = self.Dsample[i](layer_output)
            
            t_embed = self.activate(self.Time_mlp[i](t))
            t_embed = t_embed[(..., ) + (None, ) * 2]

            x = x + t_embed

        x = x.view(x.size(0), -1)
        output = self.linear(x)

        return output
    

class DiffusionDecoder(nn.Module):

    def __init__(self, t_embed_dim, output_nch=3):
        super(DiffusionDecoder, self).__init__()
        
        # hard code the hidden channels to be [16, 8, 4, 2]
        self.hidden_nchs = [1, 16, 8, 4, 2]

        self.activate = nn.GELU()

        self.t_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.GELU(),
        )

        self.Spade_block = nn.ModuleList()
        self.Time_mlp = nn.ModuleList()
        self.Usample = nn.ModuleList()

        for i in range(len(self.hidden_nchs)-1):
            self.Spade_block.append(modules.SPADE_residual_block(seg_nch=1,
                                                                 input_nch=self.hidden_nchs[i],
                                                                 output_nch=self.hidden_nchs[i+1])
                                                                 )
            self.Usample.append(modules.UpSample(input_nch=self.hidden_nchs[i+1],
                                                 output_nch=self.hidden_nchs[i+1])
                                                 )
            self.Time_mlp.append(nn.Linear(t_embed_dim, self.hidden_nchs[i+1]))

        # last layer without upsample
        self.Spade_block.append(modules.SPADE_residual_block(1, self.hidden_nchs[-1], output_nch))


    def forward(self, z, y, t):
        # sinosoidal embedding
        t = self.t_embedding(t)

        # reshape the latent style vector
        z = z.view(z.size(0), 1, 16, 16)

        for i in range(len(self.hidden_nchs)-1):
            z = self.Spade_block[i](z, y)

            t_embed = self.activate(self.Time_mlp[i](t))
            t_embed = t_embed[(..., ) + (None, ) * 2]
            
            z = self.Usample[i](z + t_embed)
        
        output = self.Spade_block[-1](z, y)
        return output


class DiffusionModel(nn.Module):

    def __init__(self, input_nch, encoder_nchs, p_size, t_embed_dim):
        super(DiffusionModel, self).__init__()

        self.encoder = DiffusionEncoder(input_nch=input_nch, 
                            hidden_nchs=encoder_nchs,
                            H=p_size[0],
                            W=p_size[1],
                            t_embed_dim=t_embed_dim)
        self.decoder = DiffusionDecoder(t_embed_dim=t_embed_dim)
    
    
    def forward(self, x, y, t):
        z = self.encoder(x, t)
        output = self.decoder(z, y, t)
        return output


if __name__ == "__main__":

    device = torch.device("cuda")
    save_path = "/home/dewei/Medical_Semantic_Diffusion/"
    
    # configuration
    batch_size = 8
    T = 100
    t_embed_dim = 32

    beta_start = 0.0001
    beta_end = 0.02
    num_timesteps = 100
    
    betas = get_beta_schedule(beta_schedule="sigmoid",
                              beta_start=beta_start,
                              beta_end=beta_end,
                              num_diffusion_timesteps=num_timesteps).to(device)

    # load model
    encoder_nchs = [4, 8, 16, 32, 64]
    p_size = [256, 256]
    model = DiffusionModel(input_nch=3, 
                           encoder_nchs=encoder_nchs, 
                           p_size=p_size, 
                           t_embed_dim=t_embed_dim).to(device)

    # simulation
    x = torch.randn((batch_size, 3, p_size[0], p_size[1]), dtype=torch.float32).to(device)
    y = torch.randn((batch_size, 1, p_size[0], p_size[1]), dtype=torch.float32).to(device)
    t = torch.randint(0, T, (batch_size,), device=device).long()
    
    sampler = DiffusionSampler(betas, device=device)
    x_t, eps = sampler.forward_sample(x, t)

    output = model(x_t, y, t)

    print(f"x_t: {x_t.dtype}, y: {y.dtype}, t: {t.dtype}, betas: {betas.dtype}")


    

    



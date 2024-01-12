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

    def __init__(self, betas, device, mode='unconditional'):
        super(DiffusionSampler, self).__init__()

        self.device = device
        self.mode = mode
        
        # given the beta schedule, compute the close-form expressions 
        self.betas = betas
        self.alphas = 1. - betas
        self.sqrt_reciprocal_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_reciprocal_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # pad 1 to the left and keep the dimension unchanged
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # variables for the type II mean expression
        self.reciprocal_one_minus_alphas_cumprod = 1.0 / (1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_alpha = torch.sqrt(self.alphas)



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


    # compute x_{t-1} from x_t
    @torch.no_grad()
    def reverse_sample(self, x_t, t, model, x_condition=None, output_type='image'):
        
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

        if self.mode == 'conditional':
            eps_pred = model(x_t.to(self.device), x_condition.to(self.device), t.to(self.device)).cpu()
        elif self.mode == 'unconditional':
            eps_pred = model(x_t.to(self.device), t.to(self.device)).cpu()
        else:
            raise ValueError

        mean = sqrt_reciprocal_alphas_t * (x_t - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t)
        std = torch.sqrt(posterior_variance_t)
        noise = torch.randn_like(x_t)

        output = mean + std * noise

        if output_type == 'image':
            return output
        elif output_type == 'gaussian':
            return mean, std, noise
        else:
            raise ValueError


    @torch.no_grad()
    def reverse_sample_typeII(self, x_t, t, model, output_type='image'):
        
        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        
        reciprocal_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.reciprocal_one_minus_alphas_cumprod, t, x_t.shape
        )

        sqrt_alphas_cumprod_prev_t = self.get_index_from_list(
            self.sqrt_alphas_cumprod_prev, t, x_t.shape
        )

        sqrt_alpha_t = self.get_index_from_list(
            self.sqrt_alpha, t, x_t.shape
        )

        alphas_cumprod_prev_t = self.get_index_from_list(
            self.alphas_cumprod_prev, t, x_t.shape
        )
        
        posterior_variance_t = self.get_index_from_list(
            self.posterior_variance, t, x_t.shape
        )

        coefficient_1 = reciprocal_one_minus_alphas_cumprod_t * sqrt_alphas_cumprod_prev_t * betas_t
        coefficient_2 = reciprocal_one_minus_alphas_cumprod_t * sqrt_alpha_t * (1. - alphas_cumprod_prev_t)
        x_0_t = self.reverse_skip(x_t, t, model)

        mean = coefficient_1 * x_0_t + coefficient_2 * x_t
        std = torch.sqrt(posterior_variance_t)
        noise = torch.randn_like(x_t)

        output = mean + std * noise

        if output_type == 'image':
            return output
        elif output_type == 'gaussian':
            return mean, std, noise
        else:
            raise ValueError

    # compute x_0 iteratively from x_t
    @torch.no_grad()
    def reverse_iterate(self, x_t, t, model, x_condition=None):
        # iterative reverse sampling
        for i in range(0, t+1):
            t_tensor = torch.full((1, ), t-i, dtype=torch.long)
            
            if self.mode == 'conditional':
                x_t = self.reverse_sample(x_t, t_tensor, model, x_condition)
            elif self.mode == 'unconditional':
                x_t = self.reverse_sample(x_t, t_tensor, model)
            else:
                raise ValueError

            x_t = torch.clamp(x_t, -1.0, 1.0)

        return x_t
    
    

    # compute x_0 directly from x_t
    @torch.no_grad()
    def reverse_skip(self, x_t, t, model, x_condition=None):

        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        sqrt_reciprocal_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape
        )
        
        if self.mode == 'conditional':
            eps_pred = model(x_t.to(self.device), x_condition.to(self.device), t.to(self.device)).cpu()
        elif self.mode == 'unconditional':
            eps_pred = model(x_t.to(self.device), t.to(self.device)).cpu()
        else:
            raise ValueError
        
        x_0_t = sqrt_reciprocal_alphas_cumprod_t * (x_t - sqrt_one_minus_alphas_cumprod_t * eps_pred)

        return x_0_t


    # compute all x_i from x_t i={t-1, ..., 0}
    @torch.no_grad()
    def serial_reverse_iterate(self, x_t, T, model, num_images, save_path, x_condition=None):
        
        step_size = int(T / num_images)

        for i in range(0, T)[::-1]:
            t_tensor = torch.full((1, ), i, dtype=torch.long)
            if self.mode == 'conditional':
                x_t = self.reverse_sample(x_t, t_tensor, model, x_condition)
            elif self.mode == 'unconditional':
                x_t = self.reverse_sample(x_t, t_tensor, model)
            else:
                raise ValueError
            x_t = torch.clamp(x_t, -1.0, 1.0)

            if i % step_size == 0:
                img = utils.tensor2pil(x_t.detach().cpu())
                utils.image_saver(np.array(img), save_path, f"step_{i}")
            

    @torch.no_grad()
    def grid_plot(self, x_t, model, save_path, x_condition=None):
        num_plot = 5
        T = len(self.betas)
        stepsize = int(T / num_plot)

        fig, axes = plt.subplots(1, 5, figsize=(16, 6))
        for i in range(0, T)[::-1]:
            t_tensor = torch.full((1, ), i, dtype=torch.long)
            
            if self.mode == 'conditional':
                x_t = self.reverse_sample(x_t, t_tensor, model, x_condition)
                x_0_t = self.reverse_skip(x_t, t_tensor, model, x_condition)
            elif self.mode == 'unconditional':
                x_t = self.reverse_sample(x_t, t_tensor, model)
                x_0_t = self.reverse_skip(x_t, t_tensor, model)
            else:
                raise ValueError
            
            x_t = torch.clamp(x_t, -1.0, 1.0)
            x_0_t = torch.clamp(x_0_t, -1.0, 1.0)

            if i % stepsize == 0:
                im_x_t = np.array(utils.tensor2pil(x_t.detach().cpu()))
                im_x_0_t = np.array(utils.tensor2pil(x_0_t.detach().cpu()))
                img = np.concatenate((im_x_t, im_x_0_t), axis=0)

                r = int(i / stepsize) // 5
                c = int(i / stepsize) % 5
                # ax = axes[r, c]
                ax = axes[c]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"denoise step = {T-i}")
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)
    

    @torch.no_grad()
    def get_conditonal_diffusion_result(self, x_t, x_condition, model):
        T = len(self.betas)
        x_0 = self.reverse_iterate(x_t, T-1, model, x_condition)
        x_0 = np.array(utils.tensor2pil(x_0.detach().cpu()))
        
        x_condition = x_condition.detach().numpy()
        x_condition = np.uint8(utils.ImageRescale(x_condition, [0, 255]))

        img = np.hstack((x_0[:, :, 0], x_condition[0, 0, :, :]))
        return img
        

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


class condition_Unet(nn.Module):

    def __init__(self,):
        super().__init__()
        image_channels = 1
        down_channels = (16, 32, 64, 128, 256)
        up_channels = (256, 128, 64, 32, 16)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        self.silu = nn.SiLU()
        
        # initial projection
        self.initial = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downwards sequential
        self.encoder_resblk = nn.ModuleList()
        self.encoder_t_linear = nn.ModuleList()
        self.encoder_down_sample = nn.ModuleList()

        for i in range(len(down_channels)-1):
            self.encoder_resblk.append(modules.residual_block(down_channels[i], down_channels[i+1]))
            self.encoder_t_linear.append(nn.Linear(time_emb_dim, down_channels[i+1]))
            self.encoder_down_sample.append(modules.DownSample(down_channels[i+1], down_channels[i+1]))
        
        # Upwards sequential
        self.decoder_spadeblk = nn.ModuleList()
        self.decoder_t_linear = nn.ModuleList()
        self.decoder_up_sample = nn.ModuleList()

        for i in range(len(up_channels)-1):
            self.decoder_spadeblk.append(modules.SPADE_residual_block(1, 2*up_channels[i], up_channels[i+1]))
            self.decoder_t_linear.append(nn.Linear(time_emb_dim, up_channels[i+1]))
            self.decoder_up_sample.append(modules.UpSample(up_channels[i], up_channels[i]))

        # final projection to output
        self.final = nn.Conv2d(up_channels[-1], out_dim, 1)

    
    def forward(self, x_t, x_condition, t):
        # time embedding
        t_embed = self.time_mlp(t)
        # initial projection 
        x = self.initial(x_t)
        
        # U-Net encoder
        features = []
        for i in range(len(self.encoder_resblk)):
            # residual block
            x = self.encoder_resblk[i](x)
            
            # add time embedding
            time_emb = self.silu(self.encoder_t_linear[i](t_embed))
            time_emb = time_emb[(..., ) + (None, ) * 2]
            x = x + time_emb
            
            features.append(x)

            # downsample
            x = self.encoder_down_sample[i](x)
        

        # U-Net decoder
        for i in range(len(self.decoder_spadeblk)):
            # upsample
            x = self.decoder_up_sample[i](x)

            # skip connection
            residual_x = features.pop()
            x = torch.cat((x, residual_x), dim=1)

            # SPADE block
            x = self.decoder_spadeblk[i](x, x_condition)
            
            # add time embedding
            time_emb = self.silu(self.decoder_t_linear[i](t_embed))
            time_emb = time_emb[(..., ) + (None, ) * 2]
            x = x + time_emb

        output = self.final(x)
        return output



    




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import unet
import diffusion_solver 
from dataloader import load_train_data



def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CelebA(root=data_path, download=True,
                                         transform=data_transform)

    test = torchvision.datasets.CelebA(root=data_path, download=True,
                                      transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])


def tensor2pil(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        b, c, h, w = image.shape
        image = image[0, :, :, :]
        if c == 1:
            image = torch.cat((image, image, image), dim=0)
        
    return reverse_transforms(image)


device = torch.device("cuda")

batch_size = 10
beta_start = 0.0001
beta_end = 0.02
T = 300
# betas = diffusion_solver.get_beta_schedule(beta_schedule="linear",
#                                             beta_start=beta_start,
#                                             beta_end=beta_end,
#                                             num_diffusion_timesteps=T)

# sampler = diffusion_solver.DiffusionSampler(betas, device=device)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.type(torch.int64))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas.to(device), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod.to(device), t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas.to(device), t, x.shape)

    eps_pred = model(x, t)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance.to(device), t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x).to(device)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image(epoch, step):
    # Sample noise
    img = torch.randn((1, 1, 256, 256), device=device)
    plt.figure(figsize=(15,3))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            plt.imshow(tensor2pil(img.detach().cpu()))
            plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path + f"epoch_{epoch}_step_{step}")


# load data
# data_path = "/home/dewei/Medical_Semantic_Diffusion/data/CelebA/"
# save_path = "/home/dewei/Medical_Semantic_Diffusion/result/example/"
# ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

# data = load_transformed_dataset()
# dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
save_path = "/home/dewei/Medical_Semantic_Diffusion/"
ckpt_path = "/home/dewei/Medical_Semantic_Diffusion/ckpt/"

with open(data_path + "OCTA_data.pickle", "rb") as handle:
    data = pickle.load(handle)

# datasets = ["drive", "stare", "chase",
#             "hrf_control", "hrf_diabetic", "hrf_glaucoma",
#             "aria_control", "aria_diabetic", "aria_amd",]
datasets = ["rose", "octa500"]
num_sample = 100
batch_size = 24
p_size = [256, 256]
intensity_range = [-1, 1]

dataloader = load_train_data(data, p_size, num_sample, datasets, intensity_range, batch_size=batch_size)


# load model
model = unet.SimpleUnet().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))


#%% ------------------------ training ------------------------
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 300 

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy.to(device), t.to(device))
    return F.l1_loss(noise.to(device), noise_pred)

for epoch in range(epochs):
    
    values = range(len(dataloader))
    with tqdm(total=len(values)) as pbar:

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # forward sample
            x0 = batch[0][:,0,:,:].unsqueeze(1)
            t = torch.randint(0, T, (x0.shape[0],)).long()

            loss = get_loss(model, x0, t)
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("epoch: %d, MSE_loss: %.4f" %(epoch, loss.item()))

            # visualize each epoch
            if step % 1000 == 0:
                sample_plot_image(epoch, step)

        scheduler.step()

    name = "diffusion_unet_OCTA.pt"
    torch.save(model.state_dict(), ckpt_path + name)



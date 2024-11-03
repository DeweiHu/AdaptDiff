import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import random
import cv2

class GaussianNoise:
    
    def __init__(self, ):
        self.mean = 0
        self.std = random.uniform(0.02, 0.15)

    def __call__(self, img):
        tensor = F.to_tensor(img)
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noise_pil = F.to_pil_image(torch.clamp(tensor + noise, 0, 1))
        return np.array(noise_pil)
    

class UnsharpMask:

    def __init__(self, kernel_size=(5, 5), sigma=1.):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.alpha = random.randint(10, 20)

    def __call__(self, img):
        im_blur = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
        im_sharp = float(self.alpha + 1) * img - float(self.alpha) * im_blur
        return np.clip(im_sharp, 0, 1)


class BrightnessAdjust:

    def __init__(self, ):
        self.magnitude = random.uniform(-0.05, 0.05)
    
    def __call__(self, img):
        im_bright = img + self.magnitude
        return np.clip(im_bright, 0, 1)


class IntensityPerturb:

    def __init__(self, ):
        self.scale = random.uniform(0.7, 0.9)
        self.bias = random.uniform(-0.01, 0.01)
    
    def __call__(self, img):
        im_perturb = self.scale * img + self.bias
        return np.clip(im_perturb, 0, 1)


def get_augment_transforms():
    transforms_list = []
    label = []

    seed = random.uniform(0, 1)

    if seed < 0.5:
        transforms_list.append(transforms.Lambda(lambda img: GaussianNoise()(img)))
        label.append("noise")

    if 0.5 <= seed < 0.8:
        transforms_list.append(transforms.Lambda(lambda img: UnsharpMask()(img)))
        label.append("unsharp")

    if 0.5 <= seed < 0.6:
        transforms_list.append(transforms.Lambda(lambda img: BrightnessAdjust()(img)))
        label.append("brightness")
    
    transforms_list.append(transforms.ToTensor())
    if 0.6 <= seed:
        transforms_list.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)))
        label.append("blur")

    transforms_list.append(transforms.Lambda(lambda img: torch.clamp(img, 0, 1)))

    return transforms.Compose(transforms_list)





if __name__ == "__main__":

    data_path = "/home/dewei/Medical_Semantic_Diffusion/data/FP_data.pickle"

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    
    dataset = "drive"
    
    im = data[dataset + "_im"][0]
    gt = data[dataset + "_gt"][0]

    im = im[1, :, :]

    for _ in range(10):
        augment_functions, label = get_augment_transforms()
        noise = augment_functions(im)
        
        print(label)
        plt.subplot(1, 2, 1), plt.imshow(im, cmap="gray")
        plt.subplot(1, 2, 2), plt.imshow(noise.numpy()[0], cmap="gray")
        plt.show()



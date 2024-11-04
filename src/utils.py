from PIL import Image
import torch
import numpy as np
import os
import cv2
from skimage.filters import threshold_multiotsu
from pynvml import *
import nibabel as nib
from torchvision import transforms


def load_binary_mask(image_path, name):
    # Load the image
    image = Image.open(os.path.join(image_path, name)).convert("L")
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Convert values from 0 and 255 to 0 and 1
    binary_array = (image_array > 0).astype(np.uint8)
    
    return binary_array


def print_gpu_utilization(stemp, visible_devices):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(visible_devices))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"{stemp}, GPU memory occupied: {info.used//1024**2} MB.")


def image_saver(img, path, name):
    img = ImageRescale(img, [0, 255]).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(os.path.join(path, "{}.png".format(name)))


def image_reader(path):
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def multi_threshold(img, num_level):
    thresholds = threshold_multiotsu(img, classes=num_level)
    alpha = np.digitize(img, bins=thresholds)
    return thresholds, alpha


def ImageRescale(im, I_range):
    im_range = im.max() - im.min()
    target_range = I_range[1] - I_range[0]
    
    if im_range == 0:
        target = np.zeros(im.shape, dtype=np.float32)
    else:
        target = I_range[0] + target_range/im_range * (im - im.min())
    return np.float32(target)

def get_ada_input_tensor(img, dpt):
    img = ImageRescale(img, [0, 1])
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    dpt = ImageRescale(dpt, [0, 1])
    dpt = torch.from_numpy(dpt[None, :, :]).unsqueeze(0)
    return img, dpt


def threshold_image(image_array):
    """Threshold the image. Pixels with intensity > 1 are set to 1."""
    thresholded = np.where(image_array > 1, 1, image_array)
    return thresholded
    
    
def nii_loader(dir):
    data_nii = nib.load(dir)
    data = np.array(data_nii.dataobj)
    return data


def nii_saver(volume,path,filename,header=None):
    output = nib.Nifti1Image(volume, np.eye(4), header=header)
    nib.save(output,os.path.join(path,filename))


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

def tensor2numpy(image):
    '''
    image: tensor with shape [c, h, w], clamped to [-1.0, 1.0]
    '''
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t[0, :, :]), 
        transforms.Lambda(lambda t: t.numpy().astype(np.float32)),
    ])
    return reverse_transforms(image)



def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def ColorSeg(pred, gt):
    h, w = pred.shape
    tn_color = np.array(Image.new('RGB', (w, h), "seashell"))
    tp_color = np.array(Image.new('RGB', (w, h), "navy"))
    fp_color = np.array(Image.new('RGB', (w, h), "limegreen"))
    fn_color = np.array(Image.new('RGB', (w, h), "crimson"))
    
    true = np.uint8(pred == gt)
    tp_mask = true * gt
    tp_mask = np.repeat(tp_mask[:, :, np.newaxis], 3, axis=2)
    
    tn_mask = true * (1-gt)
    tn_mask = np.repeat(tn_mask[:, :, np.newaxis], 3, axis=2)
    
    fp_mask = np.uint8(pred > gt)
    fp_mask = np.repeat(fp_mask[:, :, np.newaxis], 3, axis=2)
    
    fn_mask = np.uint8(pred < gt)
    fn_mask = np.repeat(fn_mask[:, :, np.newaxis], 3, axis=2)
    
    im_color = tp_color * tp_mask + tn_color * tn_mask + \
               fp_color * fp_mask + fn_color * fn_mask
    
    return im_color
from PIL import Image
import torch
import numpy as np
import os
import cv2
from skimage.filters import threshold_multiotsu
from pynvml import *
import nibabel as nib

def print_gpu_utilization(stemp, visible_devices):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(visible_devices))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"{stemp}, GPU memory occupied: {info.used//1024**2} MB.")


def image_saver(img, path, name):
    img = Image.fromarray(img.astype(np.uint8))
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


def write_depth(path, depth, bits=1, absolute_depth=False):

    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return

def _to_image_(tensor):
    img = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().astype('uint8')
    return img

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
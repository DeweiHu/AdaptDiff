import utils
import numpy as np
import imageio
import os
import pickle
from PIL import Image
from tqdm import tqdm

'''
img: normalize all input image to range [0,1], type = np.float32
    gt: binarized to 0,1, type = np.uint8
im_dir: list of directories of each image
gt_dir: list of directories of each label 

convert grayscale image to [3, H, W] by duplicating channels
'''

class pair:

    def __init__(self, im_dir, gt_dir, msk_dir=None):
        self.img_dir = im_dir
        self.gt_dir = gt_dir
        self.msk_dir = msk_dir
        
    def reader(self, root):
        fmt = root[-3:]
        if fmt == "gif":
            im = imageio.imread(root)
        else:
            im = np.array(Image.open(root))
        return im
    
    def normalize(self, im, im_type):
        if im_type == "img":
            if len(im.shape) == 3:
                opt = utils.ImageRescale(np.float32(im), [0, 1])
                opt = np.transpose(opt, (2, 0, 1))
                
                # remove transparancy channels if exists
                opt = opt[:3, :, :]

            elif len(im.shape) == 2:
                opt = utils.ImageRescale(np.float32(im), [0, 1])
                opt = np.stack([opt] * 3, axis=0)
            else:
                raise ValueError

        elif im_type == "gt":
            # remove other labels (e.g. fovea)
            opt = np.uint8(im == im.max())

            if len(opt.shape) != 2:
                opt = opt[:, :, 0]
            
        # elif im_type == "mask":
        #     opt = np.uint8(im == im.max())
        
        else:
            raise ValueError
            
        return opt
    
    def __call__(self,):
        img_list = []
        gt_list = []
        # mask_list = []
        
        for i in range(len(self.img_dir)):
            im = self.reader(self.img_dir[i])
            gt = self.reader(self.gt_dir[i])
            
            # if self.msk_dir == None:
            #     mask = np.ones([im.shape[0],im.shape[1]], dtype=np.uint8)
            # else:
            #     mask = self.reader(self.msk_dir[i])
            #     if len(mask.shape) == 3:
            #         mask = np.uint8(mask == mask.max())
            #         mask = mask[:,:,0]
                    
            img_list.append(self.normalize(im, "img"))
            gt_list.append(self.normalize(gt, "gt"))
            # mask_list.append(self.normalize(mask, "mask"))
            
        return img_list, gt_list
    

def get_paired_data(data_path):
    
    data = {}

    value = range(len(list(data_path)))
    with tqdm(total=len(value)) as pbar:
        for key in list(data_path):
            
            pbar.update(1)
            pbar.set_description(f"Processing dataset: {key}")

            img_path = data_path[key] + "img/"
            gt_path = data_path[key] + "gt/"

            img_name_list = []
            gt_name_list = []

            if key.startswith("aria"):
                for file in os.listdir(img_path):
                    if file.endswith(".tif"):
                        img_name_list.append(img_path + file)
                        name, _ = file.split(".")
                        name = name.replace(" ", "")
                        gt_name_list.append(gt_path + name + "_BDP.tif")
            else:
                for file in os.listdir(img_path):
                    img_name_list.append(img_path + file)
                for file in os.listdir(gt_path):
                    gt_name_list.append(gt_path + file)
            
            get_pair = pair(img_name_list, gt_name_list)
            data[key + "_im"], data[key + "_gt"] = get_pair()

    return data
    

if __name__ == "__main__":

    data_path = "/media/dewei/New Volume/data/"
    save_path = "/home/dewei/Medical_Semantic_Diffusion/data/"

    # fundus data 
    FP_data_path = {}

    FP_data_path["drive"] = data_path + "DRIVE/training/"
    FP_data_path["stare"] = data_path + "STARE/"
    FP_data_path["chase"] = data_path + "CHASE_DB1/"
    FP_data_path["prime_fp"] = data_path + "PRIME-FP20/prime_fundus/"

    FP_data_path["hrf_control"] = data_path + "HRF/control/"
    FP_data_path["hrf_diabetic"] = data_path + "HRF/diabetic/"
    FP_data_path["hrf_glaucoma"] = data_path + "HRF/glaucoma/"

    FP_data_path["aria_control"] = data_path + "ARIA/control/"
    FP_data_path["aria_amd"] = data_path + "ARIA/amd/"
    FP_data_path["aria_diabetic"] = data_path + "ARIA/diabetic/"

    FP_data = get_paired_data(FP_data_path)

    with open(save_path + "FP_data.pickle", "wb") as handle:
        pickle.dump(FP_data, handle)

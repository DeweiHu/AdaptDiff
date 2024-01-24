import utils
import torch
import torch.utils.data as Data
import pickle
import numpy as np
import cv2


class GetPairedDataset(Data.Dataset):

    def __init__(self, data, p_size, num_sample, datasets, intensity_range):
        super(GetPairedDataset, self).__init__()

        self.im = []
        self.gt = []

        for key in datasets:
            im_list = data[key + "_im"]
            gt_list = data[key + "_gt"]

            for i in range(len(im_list)):
                im_patch, gt_patch = self.sample(im_list[i], gt_list[i],
                                                 num_sample, p_size, 
                                                 intensity_range)
                self.im += im_patch
                self.gt += gt_patch

    def __len__(self,):
        return len(self.im)
    
    def __getitem__(self, idx):
        x = self.im[idx]
        y = self.gt[idx]
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        return x_tensor, y_tensor


    def sample(self, im, gt, num_sample, psize, intensity_range):
        dim = im.shape
        sample_x = np.random.randint(0, dim[-2]-psize[0], num_sample)
        sample_y = np.random.randint(0, dim[-1]-psize[0], num_sample)
        
        im_patch = []
        gt_patch = []
        
        for i in range(num_sample):
            px = im[:, sample_x[i]:sample_x[i]+psize[0], sample_y[i]:sample_y[i]+psize[1]]
            px = utils.ImageRescale(px, intensity_range)
            im_patch.append(px)

            py = gt[sample_x[i]:sample_x[i]+psize[0], sample_y[i]:sample_y[i]+psize[1]]
            gt_patch.append(py[None, :, :])
            
        return im_patch, gt_patch


def load_train_data(data, p_size, num_sample, datasets, intensity_range, batch_size):
    data = GetPairedDataset(data, p_size, num_sample, datasets, intensity_range)
    loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    return loader


class GetBinaryMask(Data.Dataset):

    def __init__(self, data, num_sample):  
        super(GetBinaryMask, self).__init__()  
        
        self.psize = [256, 256]
        self.crop_range = {"drive": [(100, 500), (100, 500)],
                           "stare": [(100, 500), (100, 600)]}
        self.binary_masks = []
        
        for key in self.crop_range:
            masks = data[key + "_gt"]

            r1, r2 = self.crop_range[key][0]
            c1, c2 = self.crop_range[key][1]
            
            for y in masks:
                patches = self.sample(gt=y[r1:r2, c1:c2], 
                                     num_sample=num_sample)
                self.binary_masks += patches
    
    def __len__(self, ):
        return len(self.binary_masks)

    def __getitem__(self, idx):
        y = self.binary_masks[idx]
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        return y_tensor
    
    def sample(self, gt, num_sample):
        dim = gt.shape
        sample_x = np.random.randint(0, dim[0]-self.psize[0], num_sample)
        sample_y = np.random.randint(0, dim[1]-self.psize[0], num_sample)   
        gt_patch = []
        
        for i in range(num_sample):
            py = gt[sample_x[i]:sample_x[i]+self.psize[0], \
                    sample_y[i]:sample_y[i]+self.psize[1]]
            gt_patch.append(py[None, :, :])
            
        return gt_patch


def load_binary_masks(data, num_sample, batch_size):
    masks = GetBinaryMask(data, num_sample)
    loader = Data.DataLoader(dataset=masks, 
                             batch_size=batch_size,
                             shuffle=True)
    return loader


class GetHistogramTemplate:

    def __init__(self, im_list, num_sample, intensity_range):

        self.psize = [256, 256]
        self.im = []

        for i in range(len(im_list)):
            self.im += self.sample(im_list[i], num_sample, intensity_range)

        self.len = len(self.im)

    def sample(self, im, num_sample, intensity_range):
        dim = im.shape
        sample_x = np.random.randint(0, dim[-2]-self.psize[0], num_sample)
        sample_y = np.random.randint(0, dim[-1]-self.psize[0], num_sample)
        
        im_patch = []
        
        for i in range(num_sample):
            px = im[:, \
                    sample_x[i]:sample_x[i]+self.psize[0], \
                    sample_y[i]:sample_y[i]+self.psize[1]]
            im_patch.append(utils.ImageRescale(px, intensity_range))
            
        return im_patch
    

class GetSynData(Data.Dataset):

    def __init__(self, data):
        super(GetSynData, self).__init__()
        
        self.im = []
        self.gt = []

        for x, y in data:
            self.im.append(x[None, :, :])
            self.gt.append(y[None, :, :])
        
    def __len__(self, ):
        return len(self.im)
    
    def __getitem__(self, index):
        x = self.im[index]
        y = self.gt[index]
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        y_tensor = torch.tensor(y).type(torch.int64)
        return x_tensor, y_tensor


def load_synthetic_data(data, batch_size):
    syn_data = GetSynData(data)
    loader = Data.DataLoader(dataset=syn_data, 
                             batch_size=batch_size,
                             shuffle=True)
    return loader


if __name__ == "__main__":

    data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"
    dataset = "octa500_syn.pickle"

    with open(data_path + dataset, "rb") as handle:
        data = pickle.load(handle)
    
    loader = load_synthetic_data(data, 5)

    for step, (x, y) in enumerate(loader):
        pass

    print(f"{x.shape}, {y.shape}")

    print(f"{x.max()}, {y.max()}")





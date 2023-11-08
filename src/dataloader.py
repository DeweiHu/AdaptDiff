import utils
import torch
import torch.utils.data as Data
import pickle
import numpy as np
import cv2


class get_paired_dataset(Data.Dataset):

    def __init__(self, data, p_size, num_sample, datasets, intensity_range):
        super(get_paired_dataset, self).__init__()

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
    data = get_paired_dataset(data, p_size, num_sample, datasets, intensity_range)
    loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    return loader
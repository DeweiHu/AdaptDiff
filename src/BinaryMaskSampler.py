import pickle
import numpy as np
import torch
import torch.utils.data as Data


class GetBinaryMask:

    def __init__(self, data, num_sample):    
        
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


if __name__ == "__main__":

    data_path = "/home/dewei/Medical_Semantic_Diffusion/data/"

    with open(data_path + "FP_data.pickle", "rb") as handle:
        data = pickle.load(handle)

    mask_loader = load_binary_masks(data=data, num_sample=10, batch_size=5)

    

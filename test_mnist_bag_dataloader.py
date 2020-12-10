import torch
import os
import numpy as np
from torch.utils import data 
import glob

class NumpyDataset(data.Dataset):

    def __init__(self, root_path, transforms):
        self.data_numpy_list = [x for x in glob.glob(os.path.join(root_path, '*.npy'))]
        self.transforms = transforms
        self.data_list = []
        for ind in range(len(self.data_numpy_list)):
            data_slice_file_name = self.data_numpy_list[ind]
            data_i = np.load(data_slice_file_name)
            self.data_list.append(data_i)

    def __getitem__(self, index):

        self.data = np.asarray(self.data_list[index])
        self.data = np.stack((self.data, self.data, self.data)) # gray to rgb 64x64 to 3x64x64
        if self.transforms:
            self.data = self.transforms(self.data)
        return torch.from_numpy(self.data).float()

    def __len__(self):
        return len(self.data_numpy_list)
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
from numpy.fft import fft, fft2, ifftshift, fftshift, ifft2
import torch

class CineDataset_MC(Dataset):
    def __init__(self, files, folder_path, mode, transform=None):
        super().__init__()
        self.files = files
        self.folder_path = folder_path
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.folder_path, file)
        name = file.split('.')[0]
        with h5py.File(file_path, 'r') as f:
            full_kspace = np.array(f["FullSample"])
            mask = np.array(f[f"{self.mode}_mask"])
            sense_map = np.array(f["sense_map"])
            und_kspace = np.array(f[f"{self.mode}"])
        if self.transform is not None:
            return self.transform(full_kspace), self.transform(und_kspace), mask, sense_map, name
        return full_kspace, und_kspace, mask, sense_map, name

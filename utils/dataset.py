import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from scipy import io
from glob import glob

class CoverRhoDataset(Dataset):

    def __init__(self, img_dir, rho_dir, indices = None, transform = None, repeat = 1):
        super(CoverRhoDataset, self).__init__()

        self.img_dir = img_dir
        self.indices = indices
        full_img_list = sorted(glob(self.img_dir + '/*'))
        if indices is not None:
            self.img_list = [full_img_list[i-1] for i in indices]
        else:
            self.img_list = full_img_list.copy()
        self.rho_dir = rho_dir
        full_rho_list = sorted(glob(self.rho_dir + '/*'))
        if indices is not None:
            self.rho_list = [full_rho_list[i-1] for i in indices]
        else:
            self.rho_list = full_rho_list.copy()
        if np.size(np.where(np.asarray(self.img_dir.split('/'))=='cover'))!=0:
            self.label_list = np.zeros(len(self.img_list))
        else:
            self.label_list = np.ones(len(self.img_list))
        self.len = len(self.label_list)
        self.repeat = repeat
        self.transform = transform

    def __getitem__(self, i):
        index = i % self.len
        label = np.array(self.label_list[index])
        image_path = self.img_list[index]
        img = self.transform(Image.open(image_path))
        rho_path = self.rho_list[index]
        rho = io.loadmat(rho_path)['rho']
        return img, rho, image_path.split('/')[-1], label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len

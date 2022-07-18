import os
import os.path as osp
import torch
import torch.utils.data as data
import numpy as np
import glob
import random
import cv2
from glob import glob

random.seed(1143)
def populate_train_list(images_path, mode='train'):
    train_list = [os.path.basename(f) for f in glob(os.path.join(images_path, '*'))]
    train_list.sort()
    if mode == 'train':
        random.shuffle(train_list)
    return train_list

class dataset_loader(data.Dataset):

    def __init__(self, original_folder, enhanced_folder, mode='train'):
        self.original_path = original_folder
        self.enhanced_path = enhanced_folder
        self.original_list = populate_train_list(original_folder, mode)
        self.enhanced_list = populate_train_list(enhanced_folder, mode)
        print("Total original examples:", len(self.original_list))
        print("Total enhanced examples:", len(self.enhanced_list))

    def __getitem__(self, index):
        original_filename = self.original_list[index]
        enhanced_filename = self.enhanced_list[index]

        data_original = cv2.imread(osp.join(self.original_path, original_filename), cv2.IMREAD_UNCHANGED)
        data_enhanced = cv2.imread(osp.join(self.enhanced_path, enhanced_filename), cv2.IMREAD_UNCHANGED)

        if data_original.shape[0] >= data_original.shape[1]:
            data_original = cv2.transpose(data_original)
            data_enhanced = cv2.transpose(data_enhanced)

        data_original = (np.asarray(data_original[..., ::-1]) / 255.0)
        data_enhanced = (np.asarray(data_enhanced[..., ::-1]) / 255.0)

        data_original = torch.from_numpy(data_original).float()  # float32
        data_enhanced = torch.from_numpy(data_enhanced).float()  # float32

        return data_original.permute(2, 0, 1), data_enhanced.permute(2, 0, 1)

    def __len__(self):
        return len(self.original_list)

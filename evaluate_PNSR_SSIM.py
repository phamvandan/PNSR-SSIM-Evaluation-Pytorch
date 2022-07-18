import torch
import torch.optim

import os
import argparse
import numpy as np
from utils import PSNR
from IQA_pytorch import SSIM
from data_loaders.exposure import dataset_loader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--original_folder', type=str, default="/original_folder/")
parser.add_argument('--enhanced_folder', type=str, default="/enhanced_folder/")
config = parser.parse_args()

print(config)
test_dataset = dataset_loader(original_folder=config.original_folder, enhanced_folder=config.enhanced_folder, mode='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

with torch.no_grad():
    for i, imgs in tqdm(enumerate(test_loader)):
        high_img, enhanced_img = imgs[0].cuda(), imgs[1].cuda()
        
        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)

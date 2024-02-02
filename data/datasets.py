import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from utils.positional_encodings import PosEncodingSine

# mask for outpainting    
def get_mask(img_w, img_h):
    mask = np.ones((img_w, img_h))
    index1 = int(img_w/2 + 1)
    index2 = int(3*index1 + 2)
    for j in range (img_w):
        for k in range (img_h):
            if (index1 <= k <= index2):
                mask[j, k] = 0
    return mask[np.newaxis, :, :]

def denormalize(tensors):
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([1, 1, 1])
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)

def denormalize_single(tensor):
    tensor.add_(1).mul_(0.5)
    return torch.clamp(tensor, 0, 255)
        
class ImageDataset(Dataset):
    def __init__(self, root, img_w, img_h, spe_dim):
        self.img_w = img_w
        self.img_h = img_h
        self.spe_dim = spe_dim
        self.files = sorted(glob.glob(root + "/*.*"))
        self.mask = get_mask(self.img_w, self.img_h)
        spe = PosEncodingSine(self.spe_dim, self.img_w, self.img_h, False)
        self.SPE = spe.get(8, 11)

    def __getitem__(self, index):
        gt = Image.open(self.files[index % len(self.files)])
        gt = np.asarray(gt).astype("f").transpose(2, 0, 1)/127.5-1.0 # [0, 255] to [-1, 1]
        
        masked_img = gt*(1.-self.mask)
        masked_img = torch.from_numpy(masked_img)
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(self.mask)
        combo = torch.cat([masked_img, mask])
        return {"masked_img": masked_img, "gt": gt, 'mask': mask, 'combo': combo, 'SPE': self.SPE}

    def __len__(self):
        return len(self.files)
    
    
class ImageDatasetAug(Dataset):
    def __init__(self, root, img_w, img_h, spe_dim):
        self.img_w = img_w
        self.img_h = img_h
        self.spe_dim = spe_dim
        self.files = sorted(glob.glob(root + "/*.*"))
        self.mask = get_mask(self.img_w, self.img_h)
        spe = PosEncodingSine(self.spe_dim, self.img_w, self.img_h, False)
        self.SPE = spe.get(8, 11)
        self.prob = 0.5

    def __getitem__(self, index):
        gt = Image.open(self.files[index % len(self.files)])
        gt = np.asarray(gt).astype("f")
        split_point = np.random.randint(gt.shape[1])
        left = gt[:, :split_point]
        right = gt[:, split_point:]
        gt_aug = np.concatenate((right, left), axis=1)
        
        if random.random() < self.prob:
            gt_aug = np.fliplr(gt_aug)
        
        gt_aug = gt_aug.transpose(2, 0, 1)/127.5-1.0  # [0, 255] to [-1, 1]  
        masked_img = gt_aug*(1.-self.mask)
        masked_img = torch.from_numpy(masked_img)
        gt_aug = torch.from_numpy(gt_aug)
        mask = torch.from_numpy(self.mask)
        combo = torch.cat([masked_img, mask])
        return {"masked_img": masked_img, "gt": gt_aug, 'mask': mask, 'combo': combo, 'SPE': self.SPE}

    def __len__(self):
        return len(self.files)

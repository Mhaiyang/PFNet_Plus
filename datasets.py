"""
 @Time    : 2021/7/6 10:56
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : SSI2023_PFNet_Plus
 @File    : datasets.py
 @Function: Datasets Processing
 
"""
import os
import os.path
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]
    # img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.png')]
    # return [(os.path.join(image_path, img_name + '.png'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def collate(self, batch):
        size = [352, 384, 416, 448, 480][np.random.randint(0, 5)]
        # size = [384, 416, 448, 480][np.random.randint(0, 4)]
        # size = [416, 448, 480][np.random.randint(0, 3)]
        image, mask = [list(item) for item in zip(*batch)]

        image = torch.stack(image, dim=0)
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=True)
        mask = torch.stack(mask, dim=0)
        mask = F.interpolate(mask, size=(size, size), mode="nearest")

        return image, mask

    def __len__(self):
        return len(self.imgs)


"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : SSI2023_PFNet_Plus
 @File    : config.py
 @Function: Configuration
 
"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'
# backbone_path = './backbone/resnext/resnext_101_32x4d.pth'
# backbone_path = './backbone/Conformer_base_patch16.pth'

# COS
# datasets_root = '/home/iccd/data/NEW'
datasets_root = '/media/iccd/disk/NEW'
# datasets_root = '/media/iccd/disk2/data/NEW'

cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')

# polyp segmentation
medical_training_root = '/home/iccd/data/medical_new/train'
CVC_300_path = '/home/iccd/data/medical_new/test/CVC-300'
CVC_ClinicDB_path = '/home/iccd/data/medical_new/test/CVC-ClinicDB'
CVC_ColonDB_path = '/home/iccd/data/medical_new/test/CVC-ColonDB'
ETIS_LaribPolypDB_path = '/home/iccd/data/medical_new/test/ETIS-LaribPolypDB'
Kvasir_path = '/home/iccd/data/medical_new/test/Kvasir'


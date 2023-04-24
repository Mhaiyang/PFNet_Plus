"""
 @Time    : 2021/7/6 14:36
 @Author  : Haiyang Mei
 @E-mail  : haiyang.mei@outlook.com
 
 @Project : SSI2023_PFNet_Plus
 @File    : infer.py
 @Function: Inference
 
"""
import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from config import *
from misc import *
from model.v33 import V33

torch.manual_seed(2023)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

# ckpt_path = '/media/iccd/disk2/cos_ckpt'
# results_path = '/media/iccd/disk2/cos_results'
ckpt_path = '/media/iccd/disk/COS/cos_ckpt'
results_path = '/media/iccd/disk/COS/cos_results'
# ckpt_path = '/media/iccd/disk2/cos_glass_ckpt'
# results_path = '/media/iccd/disk2/cos_glass_results'
# ckpt_path = '/home/iccd/cos/extension/final/ckpt'
# results_path = '/home/iccd/cos/extension/final/results'
check_mkdir(results_path)

exp_name = 'V33_33'
args = {
    'snapshot': '60',
    'scale': 480,
    'save_results': True
}


print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       ('CHAMELEON', chameleon_path),
                       ('CAMO', camo_path),
                       ('COD10K', cod10k_path),
                       ])

results = OrderedDict()

def main():
    net = Ablation_BCEnPMFM(backbone_path).cuda(device_ids[0])

    if len(args['snapshot']) > 0:
        print('Load snapshot {} for testing'.format(args['snapshot']))
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        print('Load {} succeed!'.format(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'image')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                start_each = time.time()
                _, _, _, prediction = net(img_var)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, exp_name, name, img_name + '.png'))
            print(('{} {}'.format(exp_name, args['snapshot'])))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()

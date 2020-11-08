import os
import numpy as np
from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from dataset import transforms as mytrans
import myutils


class PreTrain_DS(data.Dataset):

    def __init__(self, root, output_size, dataset_file='./assets/pretrain.txt', clip_n=3, max_obj_n=11):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        self.img_list = list()
        self.mask_list = list()

        dataset_list = list()
        with open(os.path.join(dataset_file), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()

                img_dir = os.path.join(root, 'JPEGImages', dataset_name)
                mask_dir = os.path.join(root, 'Annotations', dataset_name)

                img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + sorted(glob(os.path.join(img_dir, '*.png')))
                mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

                if len(img_list) > 0:
                    if len(img_list) == len(mask_list):
                        dataset_list.append(dataset_name)
                        self.img_list += img_list
                        self.mask_list += mask_list
                        print(f'\t{dataset_name}: {len(img_list)} imgs.')
                    else:
                        print(f'\tPreTrain dataset {dataset_name} has {len(img_list)} imgs and {len(mask_list)} annots. Not match! Skip.')
                else:
                    print(f'\tPreTrain dataset {dataset_name} doesn\'t exist. Skip.')

        print(myutils.gct(), f'{len(self.img_list)} imgs are used for PreTrain. They are from {dataset_list}.')

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = mytrans.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_pil = myutils.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = myutils.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        for i in range(self.clip_n):
            img, mask = img_pil, mask_pil
            if i > 0:
                img, mask = self.random_horizontal_flip(img, mask)
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            img, mask = self.random_resize_crop(img, mask)

            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            frames[i] = self.to_tensor(img)
            masks[i] = mask

        info = {
            'name': self.img_list[idx]
        }
        return frames, masks[:, :obj_n], obj_n, info


if __name__ == '__main__':
    pass

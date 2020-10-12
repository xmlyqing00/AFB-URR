import os
import random
import numpy as np
from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from dataset import transforms as mytrans
import myutils


class DAVIS_Train_DS(data.Dataset):

    def __init__(self, root, output_size, imset='2017/train.txt', clip_n=3, max_obj_n=11):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        dataset_path = os.path.join(root, 'ImageSets', imset)
        self.dataset_list = list()
        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1), (0.95, 1.05))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        idx_list = list(range(len(img_list)))
        random.shuffle(idx_list)
        idx_list = idx_list[:self.clip_n]

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        for i, frame_idx in enumerate(idx_list):
            img = myutils.load_image_in_PIL(img_list[frame_idx], 'RGB')
            mask = myutils.load_image_in_PIL(mask_list[frame_idx], 'P')

            if i > 0:
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            roi_cnt = 0
            while roi_cnt < 10:
                img_roi, mask_roi = self.random_resize_crop(img, mask)

                mask_roi = np.array(mask_roi, np.uint8)

                if i == 0:
                    mask_roi, obj_list = self.to_onehot(mask_roi)
                    obj_n = len(obj_list) + 1
                else:
                    mask_roi, _ = self.to_onehot(mask_roi, obj_list)

                if torch.any(mask_roi[0] == 0).item():
                    break

                roi_cnt += 1

            frames[i] = self.to_tensor(img_roi)
            masks[i] = mask_roi

        info = {
            'name': video_name,
            'idx_list': idx_list
        }

        return frames, masks[:, :obj_n], obj_n, info


class DAVIS_Test_DS(data.Dataset):

    def __init__(self, root, img_set='2017/val.txt', max_obj_n=11, single_obj=False):
        self.root = root
        self.single_obj = single_obj
        dataset_path = os.path.join(root, 'ImageSets', img_set)
        self.dataset_list = list()

        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        first_mask = myutils.load_image_in_PIL(mask_list[0], 'P')
        first_mask_np = np.array(first_mask, np.uint8)

        if self.single_obj:
            first_mask_np[first_mask_np > 1] = 1

        h, w = first_mask_np.shape
        obj_n = first_mask_np.max() + 1
        video_len = len(img_list)

        frames = torch.zeros((video_len, 3, h, w), dtype=torch.float)
        masks = torch.zeros((1, obj_n, h, w), dtype=torch.float)

        mask, _ = self.to_onehot(first_mask_np)
        masks[0] = mask[:obj_n]

        for i in range(video_len):
            img = myutils.load_image_in_PIL(img_list[i], 'RGB')
            frames[i] = self.to_tensor(img)

        info = {
            'name': video_name,
            'num_frames': video_len,
        }

        return frames, masks, obj_n, info


if __name__ == '__main__':
    pass

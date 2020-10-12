import os
import random
import json
import numpy as np
from glob import glob
from itertools import compress

import torch
from torch.utils import data
import torchvision.transforms as TF

from dataset import transforms as mytrans
import myutils


class YouTube_Train_DS(data.Dataset):

    def __init__(self, root, output_size, dataset_file='meta.json', clip_n=3, max_obj_n=11):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            meta_data = json.load(json_file)

        self.dataset_list = list(meta_data['videos'])
        self.dataset_size = len(self.dataset_list)

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.3, 0.5), (0.95, 1.05))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

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


class YouTube_Test_DS(data.Dataset):

    def __init__(self, root, dataset_file='meta.json', output_size=(495, 880), max_obj_n=11):
        self.root = root
        self.max_obj_n = max_obj_n
        self.out_h, self.out_w = output_size

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.dataset_list = list(self.meta_data['videos'])
        self.dataset_size = len(self.dataset_list)

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        basename_list = [os.path.basename(x)[:-4] for x in img_list]
        video_len = len(img_list)
        selected_idx = np.ones(video_len, np.bool)

        objs = self.meta_data['videos'][video_name]['objects']
        obj_n = 1
        video_obj_appear_st_idx = video_len

        for obj_idx, obj_gt in objs.items():
            obj_n = max(obj_n, int(obj_idx) + 1)
            video_obj_appear_idx = basename_list.index(obj_gt['frames'][0])
            video_obj_appear_st_idx = min(video_obj_appear_st_idx, video_obj_appear_idx)

        selected_idx[:video_obj_appear_st_idx] = False
        selected_idx = selected_idx.tolist()

        img_list = list(compress(img_list, selected_idx))
        basename_list = list(compress(basename_list, selected_idx))

        video_len = len(img_list)
        obj_vis = np.zeros((video_len, obj_n), np.uint8)
        obj_vis[:, 0] = 1
        obj_st = np.zeros(obj_n, np.uint8)

        tmp_img = myutils.load_image_in_PIL(img_list[0], 'RGB')
        original_w, original_h = tmp_img.size
        if original_h < self.out_h:
            out_h, out_w = original_h, original_w
        else:
            out_h = self.out_h
            out_w = int(original_w / original_h * self.out_h)
        masks = torch.zeros((obj_n, out_h, out_w), dtype=torch.bool)

        basename_to_save = list()
        for obj_idx, obj_gt in objs.items():
            obj_idx = int(obj_idx)
            basename_to_save += obj_gt['frames']

            frame_idx = basename_list.index(obj_gt['frames'][0])
            obj_st[obj_idx] = frame_idx
            obj_vis[frame_idx:, obj_idx] = 1

            mask_path = os.path.join(mask_dir, obj_gt['frames'][0] + '.png')
            mask_raw = myutils.load_image_in_PIL(mask_path, 'P')
            mask_raw = mask_raw.resize((out_w, out_h))
            mask_raw = torch.from_numpy(np.array(mask_raw, np.uint8))

            masks[obj_idx, mask_raw == obj_idx] = 1

        basename_to_save = sorted(list(set(basename_to_save)))

        frames = torch.zeros((video_len, 3, out_h, out_w), dtype=torch.float)
        for i in range(video_len):
            img = myutils.load_image_in_PIL(img_list[i], 'RGB')
            img = img.resize((out_w, out_h))
            frames[i] = self.to_tensor(img)

        info = {
            'name': video_name,
            'num_frames': video_len,
            'obj_vis': obj_vis,
            'obj_st': obj_st,
            'basename_list': basename_list,
            'basename_to_save': basename_to_save,
            'original_size': (original_h, original_w)
        }

        return frames, masks, obj_n, info


if __name__ == '__main__':
    pass

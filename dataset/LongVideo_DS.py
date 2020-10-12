import os
import numpy as np
from glob import glob

from torch.utils import data
import torchvision.transforms as TF

from dataset import transforms as mytrans
import myutils


class LongVideo_Test_DS(data.Dataset):

    def __init__(self, root, dataset_file='videos.txt', max_obj_n=5):

        self.root = root
        self.dataset_list = list()

        with open(os.path.join(root, dataset_file), 'r') as lines:
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
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        return video_name, img_dir, mask_dir


class Video_DS(data.Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_list = sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png')))
        self.mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        first_mask = myutils.load_image_in_PIL(self.mask_list[0], 'P')
        first_frame = myutils.load_image_in_PIL(self.img_list[0])
        self.first_name = os.path.basename(self.img_list[0])[:-4]

        first_mask = np.array(first_mask, np.uint8)
        self.obj_n = first_mask.max() + 1
        self.img_list = self.img_list[1:]
        self.video_len = len(self.img_list)

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(self.obj_n, shuffle=False)

        first_mask, _ = self.to_onehot(first_mask)
        self.first_frame = self.to_tensor(first_frame)
        self.first_mask = first_mask[:self.obj_n]

    def __len__(self):
        return self.video_len

    def __getitem__(self, idx):
        img = myutils.load_image_in_PIL(self.img_list[idx], 'RGB')
        frame = self.to_tensor(img)
        img_name = os.path.basename(self.img_list[idx])[:-4]

        return frame, img_name


if __name__ == '__main__':
    pass

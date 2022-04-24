import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import  Image


class LongVideoDataset(object):
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://www.kaggle.com/datasets/gvclsu/long-videos'
    VOID_LABEL = 255

    def __init__(self, root_folder, task='semi-supervised', sequences='all', filelist=None, resolution=(640, 480)):
        """
        Class to read the DAVIS dataset
        :param root_folder: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.root_folder = root_folder
        self.resolution = resolution
        self.img_path = os.path.join(self.root_folder, 'JPEGImages')
        self.mask_path = os.path.join(self.root_folder, 'Annotations')

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.root_folder, filelist), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        while True:
            try:
                sequences_names.remove('')
            except ValueError as e:
                break
        print('Seq names:', sequences_names)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0:
                images = np.sort(glob(os.path.join(self.img_path, seq, '*.png'))).tolist()
                if len(images) == 0:
                    raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            self.sequences[seq]['masks'] = masks
            self.sequences[seq]['images'] = masks
        
    def _check_directories(self):
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(f'WaterDataset not found in the specified directory, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))

            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from .base import BaseDataset


class VOCLocKxSegmentation(BaseDataset):
    CLASSES = [
        '1', '2', '3', '4', '5', '6'
    ]
    print("VOCLkx")
    NUM_CLASS = 7
    IN_CHANNELS = 3
    CROP_SIZE = 256
    BASE_DIR = './input/'
    CLASS_WEIGHTS = None
    INPUT = '/kaggle/input'
    def __init__(self, root='/input/', split='train', mode=None):

        super(VOCLocKxSegmentation, self).__init__(root, split, mode)
        _voc_root = os.path.join(self.BASE_DIR)
        train_path = os.path.join(_voc_root, 'data_train.npz')
        labels_path = os.path.join(_voc_root, 'labels_train.npz')
        m_train_dataset = np.load( self.INPUT + '/seismic-facies/data_train.npz', allow_pickle=True, mmap_mode='r')['data']
        m_train_labels = np.load(self.INPUT + '/seismic-facies/labels_train.npz', allow_pickle=True, mmap_mode='r')['labels']

        m_train_dataset1, m_test_dataset, m_train_labels1, m_test_labels = train_test_split(
        m_train_dataset, m_train_labels, test_size = 0.2, random_state = 42)

        print("Run NAS UNet from LOC KX")
        self.joint_transform = None

        if self.mode == 'train':
            self.train_dataset = m_train_dataset1
            self.train_labels = m_train_labels1
            print(f"train sizes: +data={len(self.train_dataset)}, label={len(self.train_labels)}")
            print(f"test sizes: + data={len(m_test_dataset)}, label={len(m_test_labels)}")
        elif self.mode == 'val':
            self.train_dataset = m_test_dataset
            self.train_labels = m_test_labels
            print(f"train val size: + data={len(m_test_dataset)}, label={len(m_test_labels)}")
            print(f"test val size: + data={len(m_test_dataset)}, label={len(m_test_labels)}")
            # print(f"hê")
        # elif self.mode == 'test':
        #     self.images = []
        #     return
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        #img = self.train_dataset[:, :, index]
        #label = self.train_labels[:, :, index]
        img = self.train_dataset[:, index, :]
        label = self.train_labels[:, index, :]
        img = np.expand_dims(img, axis=2).astype('float32')
        label = np.expand_dims(label, axis=2).astype('float32')

        img = cv2.resize(img, (128, 128))
        label = cv2.resize(label, (128, 128))

        img = img / np.amax(img)
        img = np.clip(img, 0, 255)
        img = (img * 255).astype(int)
        img = img / 255.
        #-----------------------------------------------------
        # img = (img - np.min(img)) / ( (np.max(img) - np.min(img)))
        # #-----------------------------------------------------
        img = np.append(np.append([img], [img], axis=0), [img], axis=0)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return img, label

    def __len__(self):
        return len(self.train_dataset[0])
        #return len(self.train_dataset[0][0])
    @property
    def pred_offset(self):
        return 0

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
import os


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        # self.scale = scale
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = []
        files = listdir(imgs_dir)
        files.sort()
        for file in files:
            sub_img_dir = os.path.join(imgs_dir, file)
            image_files = listdir(sub_img_dir)
            image_files.sort()
            sub_ids = [os.path.join(file, image_f) for image_f in image_files]
            self.ids = self.ids + sub_ids
    
        # for file in listdir(imgs_dir)
        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #             if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_image(cls, img):
        img = 255 - img
        img[np.where(img == 255)] = 0
        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), 3, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)
        
        #cv2.imwrite('img1.png', img)
        img_new = np.array([img[:, :, 0]])
        #print(img_new.shape)
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        # img_nd = np.array(pil_img)

        # if len(img_nd.shape) == 2:
        #     img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        #img_new = img_new.transpose((2, 0, 1))
        if img_new.max() > 1:
            img_new = img_new / 255

        return img_new
    
    # @classmethod
    # def dilat_img(cls, img, mask):
    #     valid_point = np.logical_or.reduce(mask > 0, axis=2)
    #     valid_list = np.argwhere(valid_point==True)
    #     for dot in valid_point:
    #         cv2.
    #     img_new[np.where(valid_point)] = 4 

    
    @classmethod
    def preprocess_mask(cls, img, ignore_mask):
        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        #print(point_list.shape)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), 3, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)

        #cv2.imwrite('mask1.png', img)
        valid_point = np.logical_or.reduce(img > 0, axis=2)
        new_img = np.zeros((img.shape[0], img.shape[1]),dtype=np.float32)
        new_img[np.where(valid_point)] = 1
        new_img[np.where(ignore_mask)] = -1

        # img[np.where(ignore_mask)] = -1
        # img_b = img[:, :, 0]
        # img_g = img[:, :, 1]
        # img_r = img[:, :, 2]

        # if img_b.max() > 0 and img_g.max() > 0 and img_r.max() > 0:
        #     valid_point = np.logical_or.reduce(img_b > 0, axis=1)
        #     img_new = img_b
        #     img_new[np.where(valid_point)] = 1
        # elif img_b.max() > 0 :
        #     valid_point = np.logical_or.reduce(img_b > 0, axis=1)
        #     img_new = img_b
        #     img_new[np.where(valid_point)] = 1
        # elif img_g.max() > 0 :
        #     valid_point = np.logical_or.reduce(img_g > 0, axis=1)
        #     img_new = img_g
        #     img_new[np.where(valid_point)] = 1
        # elif img_r.max() > 0 :
        #     valid_point = np.logical_or.reduce(img_r > 0, axis=1)
        #     img_new = img_r
        #     img_new[np.where(valid_point)] = 1
        
        # img_new = img_new.transpose((2, 0, 1))
        # if img_new.max() > 1:
        #     img_new = img_new / 255

        return new_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx)
        img_file = glob(self.imgs_dir + idx)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = cv2.imread(mask_file[0])
        img = cv2.imread(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess_image(img)
        #ignore_mask = np.logical_or.reduce(img == 0, axis=2)
        ignore_mask = img[0]==0
        mask = self.preprocess_mask(mask, ignore_mask)

        # 1/2 image
        _, h, w = img.shape
        img = img[:, int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]
        mask = mask[int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'ignore_mask': torch.from_numpy(ignore_mask)}

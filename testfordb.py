import numpy as np 
import cv2
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import os
import torch

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

path = 'test_db'
mkdir(path)

dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/t_pc/'
dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/instance_segment/'
#nn = '000000.png'
batch_size = 1
dataset = BasicDataset(dir_img, dir_mask)
# n_val = int(len(dataset) * val_percent)
# n_train = len(dataset) - n_val
# train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
# val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

for index, bach in enumerate(train_loader):
    img = bach['image']
    mask = bach['mask']
    # print(index, img.shape, mask.shape)
    img = img[0]
    #mmask = mask[0]
    img = img.data.numpy()
    mask = mask.data.numpy()
    print(img.shape, mask.shape)
    img = np.transpose(img, (1, 2, 0)) * 255.
    mask = np.transpose(mask, (1, 2, 0))
    mask[np.where(mask == 1)] = 255
    mask[np.where(mask == 0)] = 155
    mask[np.where(mask == -1)] = 0
    cv2.imwrite(os.path.join(path, str(index) + '_img.jpg'), img)
    cv2.imwrite(os.path.join(path, str(index) + '_mask.jpg'), mask)
    if index>10:
        break

# mask = cv2.imread(dir_img + nn)
# img = cv2.imread(dir_mask + nn)

# img = bd.preprocess_image(img)
# ignore_mask = img[0]==0
# mask = bd.preprocess_mask(mask, ignore_mask)

# out_img = 255 * np.hstack((img,img,img)).astype(int)

# print(out_img.max(), out_img.min(), out_img.shape)
# cv2.imwrite('img1.jpg', out_img)
# cv2.imwrite('mask1.png', 255 * np.dstack((mask,mask,mask)))

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train','val', 'test', 'trainsemi', 'loadsemi']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)


    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)

    elif mode == 'test':
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)
    
    elif mode == 'trainsemi': #On prend nos nouvelles images et leurs pseudolabels + les images de train habituelles.
        trainsemi_img_path = os.path.join(root, 'train', 'Img-Unlabeled')
        trainsemi_mask_path = os.path.join(root, 'train', 'GTsemi')
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')

        images_semi = os.listdir(trainsemi_img_path)
        labels_semi = os.listdir(trainsemi_mask_path)
        images_train = os.listdir(train_img_path)
        labels_train = os.listdir(train_mask_path)

        images_semi.sort()
        labels_semi.sort()
        images_train.sort()
        labels_train.sort()

        #On mélange les images semi-supervisées et les images de train
        for it_im, it_gt in zip(images_semi, labels_semi):
            item = (os.path.join(trainsemi_img_path, it_im), os.path.join(trainsemi_mask_path, it_gt))
            items.append(item)
        
        for it_im, it_gt in zip(images_train, labels_train):
            item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)
        

    elif mode == 'loadsemi':
        loadsemi_img_path = os.path.join(root, 'train', 'Img-Unlabeled')

        images = os.listdir(loadsemi_img_path)

        images.sort()

        for it_im in images:
            item = (os.path.join(loadsemi_img_path, it_im), os.path.join(loadsemi_img_path, it_im))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert('L')

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return [img, mask, img_path]
    

class MyDataloader(object):
    def __init__(self,args):
        self.args = args
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.20))
        ])
    
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def create_labelled_dataloaders(self):
        if self.args.augment is True:
            #TODO
            ...
        train_set = MedicalImageDataset('train',
                                        self.args.root_dir,
                                        transform=self.transform,
                                        mask_transform=self.mask_transform,
                                        augment=self.args.augment,
                                        equalize=False)

        train_loader = DataLoader(train_set,
                                batch_size=self.args.batch_size,
                                worker_init_fn=np.random.seed(0),
                                num_workers=self.args.num_workers,
                                shuffle=True)
        
        val_set = MedicalImageDataset('val',
                                    self.args.root_dir,
                                    transform=self.transform,
                                    mask_transform=self.mask_transform,
                                    equalize=False)
        
        val_loader = DataLoader(val_set,
                                batch_size=self.args.val_batch_size,
                                worker_init_fn=np.random.seed(0),
                                num_workers=self.args.num_workers,
                                shuffle=True)
        
        test_set = MedicalImageDataset('test',
                                    self.args.root_dir,
                                    transform=self.transform,
                                    mask_transform=self.mask_transform,
                                    equalize=False)
        
        test_loader = DataLoader(test_set,
                                batch_size=1,
                                worker_init_fn=np.random.seed(0),
                                num_workers=5,
                                shuffle=False)
        
        return train_loader, val_loader, test_loader 
    
    def create_unlabelled_dataloaders(self):
        unlabelled_set =  MedicalImageDataset('trainsemi',
                                    self.args.root_dir,
                                    transform=self.transform,
                                    mask_transform=self.mask_transform,
                                    equalize=False)
        unlabelled_loader = DataLoader(unlabelled_set,
                                batch_size=self.args.batch_size,
                                worker_init_fn=np.random.seed(0),
                                num_workers=self.args.num_workers,
                                shuffle=True) #On veut les shuffle ici pour que les images semi-supervisées soient mélangées avec les images de train.
        
        return unlabelled_loader

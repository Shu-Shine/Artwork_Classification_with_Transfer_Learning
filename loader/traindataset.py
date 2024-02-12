import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torch.autograd import Variable as V

import numpy as np
import torchvision as tv
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class ADataset(Dataset):
    def __init__(self, data, root_dir='', mode='train'): 
        self.root_dir = root_dir

        self.data = data
        self.mode = mode  # for training or validation dataset

        # create two different transforms
        if mode == 'train':
            self._transform = tv.transforms.Compose([

                transforms.RandomRotation(degrees=(0, 15), expand=True),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(p=0.3),

                # transforms.RandomResizedCrop(224),
                # transforms.RandomCrop(224, padding=64, padding_mode='symmetric'),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,

                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
            ])
        elif mode == 'val':
            self._transform = tv.transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]))#, f'image_{idx}.jpg'
        # if the image not exist, return
        image = Image.open(img_name).convert('RGB')

        label = self.data.iloc[idx, 1]  

        if self._transform:
            img = self._transform(image)
            image = V(img)

        return image, label

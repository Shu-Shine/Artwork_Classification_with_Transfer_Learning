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
    def __init__(self, csv_file, root_dir='./imgs', transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 0]))#, f'image_{idx}.jpg'
        image = Image.open(img_name).convert('RGB')

        label = self.data.iloc[idx, 1]  # Assuming the label is in the first column

        if self.transform:
            image = self.transform(image)
            image = V(image)


        return image, label

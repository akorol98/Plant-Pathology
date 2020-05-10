
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from skimage.transform import rotate
from skimage.transform import resize
import torchvision.transforms as transforms


class PlantPathologyDataset(Dataset):
    """Plant Pathology dataset."""

    def __init__(self, csv_file, root_dir, transform=None, slice=None, data_type='train'):
        """
        Args:
            csv_file (string): Path to the csv file with image discription.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if slice:
            type, idx = slice.split(' ')[0], int(slice.split(' ')[1])

            if type == 'before':
                self.labels = pd.read_csv(csv_file)[:idx]
            elif type == 'after':
                self.labels = pd.read_csv(csv_file)[idx:]
        else:
            self.labels = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.image_id[idx]+'.jpg')
        image = io.imread(img_name)

        if self.data_type == 'train':
            healthy = self.labels.healthy[idx]
            multiple_diseases = self.labels.multiple_diseases[idx]
            rust = self.labels.rust[idx]
            scab = self.labels.scab[idx]

            sample = {'image': image,
                      'labels': np.array([healthy,
                                          multiple_diseases,
                                          rust,
                                          scab
                                         ])}
        else:
            sample = {'image': image}


        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 2:
            image, labels = sample['image'], sample['labels']
        else:
            image = sample['image']

        # swap color axis
        image = resize(image, (700, 700))
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).float()
#         in_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                 std=[0.229, 0.224, 0.225])])
        in_transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
        image = in_transform(image)

        if len(sample) == 2:
            return {'image': image,
                    'labels': torch.from_numpy(labels).float()}
        else:
            return {'image': image}


import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from skimage.transform import rotate

class PlantPathologyDataset(Dataset):
    """Plant Pathology dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image discription.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.image_id[idx]+'.jpg')
        image = io.imread(img_name)
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

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # rotate if needed
        if image.shape[0] == 2048:
            image = image.transpose((2, 1, 0))
        else:
            # swap color axis
            image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

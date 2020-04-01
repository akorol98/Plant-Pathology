import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from skimage import transform
from skimage import img_as_ubyte


class DataAugmentation:
    def __init__(self, DATA_PATH, max_rot_angle):
        self.DATA_PATH = DATA_PATH
        self.max_rot_angle = max_rot_angle



    def generate(self, path_to_save = None, pref = None):
        if not path_to_save:
            path_to_save = self.DATA_PATH
        if pref:
            pref = pref + '_'
        else:
            pref = ''

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        for image in os.listdir(self.DATA_PATH):
            if image.split('.')[1] != 'jpg':
                continue

            im = io.imread(self.DATA_PATH + image)
            ## Random flip angle
            angle = np.random.uniform(0, self.max_rot_angle)
            im = transform.rotate(im, angle)
            io.imsave(os.path.join(path_to_save,  pref + image), img_as_ubyte(im))

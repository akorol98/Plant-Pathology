import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io
from skimage import transform
from skimage import img_as_ubyte


class DataAugmentation:
    def __init__(self, DATA_PATH, im_shape=(224,224)):
        self.DATA_PATH = DATA_PATH
        self.im_shape = im_shape



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
                
            im_number = image.split('.')[0].split('_')[1] + '_'

            im = io.imread(self.DATA_PATH + image)
            
            ## Rotate and flip
            angles = [0, 90, 180, 270]
            
            for i,angle in enumerate(angles):
                
                im = transform.resize(im, self.im_shape)
                im = transform.rotate(im, angle)
                
                io.imsave(os.path.join(path_to_save,  'Train_'+im_number+str(i)+'.jpg'), img_as_ubyte(im))
                
                im = im[:, ::-1]
                io.imsave(os.path.join(path_to_save,  'Train_'+im_number+str(i+4)+'.jpg'), img_as_ubyte(im))
                
            
            

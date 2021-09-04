import os
import glob

import numpy as np
from torch.utils.data import Dataset
from .cvtransforms import *
import torch



class LRWDataset(Dataset):
    def __init__(self, phase, args):

        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        
        if(not hasattr(self.args, 'is_aug')):
            setattr(self.args, 'is_aug', True)

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('lrw_roi_80_116_175_211_npy_gray_pkl_jpeg', label, phase, '*.pkl'))                    
            files = sorted(files)
            self.list += [file for file in files]
        
    def __getitem__(self, idx):
        tensor = torch.load(self.list[idx])                    
        
        inputs = tensor.get('video')
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:,:,:,0]
        
        if self.phase == 'train':
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
        
        result = {}            
        result['video'] = torch.FloatTensor(batch_img[:,np.newaxis,...])
        #print(result['video'].size())
        result['label'] = tensor.get('label')
        result['duration'] = 1.0 * tensor.get('duration')

        return result

    def __len__(self):
        return len(self.list)


class AVDataset(Dataset):
    def __init__(self, data_dir, phase):
        self.list = []
        self.phase = phase        

        self.list = sorted(glob.glob(os.path.join(data_dir, phase, '*.pkl')))

    def __getitem__(self, idx):
        tensor = torch.load(self.list[idx])                    
        
        inputs = tensor.get('video')
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :]
        
        if self.phase == 'train':
            batch_img = RandomCrop(inputs, (88, 88))
            batch_img = HorizontalFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(inputs, (88, 88))
        
        result = {}            
        result['video'] = torch.FloatTensor(batch_img[:, np.newaxis,...])
        result['label'] = tensor.get('label')
        result['duration'] = 1.0 * tensor.get('duration')

        return result

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    import cv2
    dataset = AVDataset('datasets/avletters_digits_npy_gray_pkl_jpeg', 'train')
    for batch in dataset:
        break
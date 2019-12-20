import os,sys, torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np
import random


class CrossedLabelDataset(Dataset):
    def __init__(self, label_file, root_dir, label_idx=1, transform=None):
        self.data = pd.read_csv(label_file)
        self.transform = transform
        self.label_idx = label_idx
        self.root_dir = root_dir
        self.data_map = []
        self.n_data = self.data.shape[0]
        all_cat = self.data.iloc[:,label_idx]
        self.categories = all_cat.unique()
        self.n_category = len(self.categories)
        tmp_map = {}
        for i_cat in range(self.n_category):
            my_cat = self.categories[i_cat]
            tmp_map[my_cat]=self.data.index[all_cat==my_cat].tolist()

        print('%d/%d has been processed.' % (0,self.n_data),end='')
        for i in range(self.n_data):
            self.data_map.append([[], [], 0, 0])
            my_cls = self.data.iloc[i,label_idx]
            for data_idx in tmp_map[my_cls]:
                if data_idx==i:
                    continue
                self.data_map[i][1].append(data_idx)
            for cls_idx in tmp_map.keys():
                if cls_idx==my_cls:
                    continue
                else:
                    for data_idx in tmp_map[my_cls]:
                        self.data_map[i][0].append(data_idx)
            if (i+1)%100==0:
                print('\r%d/%d has been processed.'%(i+1,self.n_data),end='')

        print('\r%d%d has been processed.' % (i+1,self.n_data), end='')
        print('')

    def printMe(self):
        print(self.data_map)

    def shufflePairs(self, at_record, target_list_at):
        random.shuffle(self.data_map[at_record][target_list_at])

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx,0])
        my_f_sample_count = self.data_map[idx][2]
        if my_f_sample_count>=len(self.data_map[idx][0]):
            my_f_sample_count = 0
            self.shufflePairs(idx,0)
        my_t_sample_count = self.data_map[idx][3]
        if my_t_sample_count>=len(self.data_map[idx][1]):
            my_t_sample_count = 0
            self.shufflePairs(idx,1)
        t_img_name = os.path.join(self.root_dir, self.data.iloc[self.data_map[idx][1][my_t_sample_count], 0])
        f_img_name = os.path.join(self.root_dir, self.data.iloc[self.data_map[idx][2][my_f_sample_count], 0])
        self.data_map[idx][2] = my_f_sample_count + 1
        self.data_map[idx][3] = my_t_sample_count + 1
        img = io.imread(img_name)
        timg = io.imread(t_img_name)
        fimg = io.imread(f_img_name)

        sample = {'im':img, 'true_im':timg, 'false_im':fimg}
        if self.tranform:
            sample = self.transform(sample)

        return sample
import os,sys, torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import numpy as np
import random


class CrossedLabelDataset(Dataset):
    def __init__(self, label_file, root_dir, label_idx=1, transform=None, processed_path=None):
        self.data = pd.read_csv(label_file)
        self.transform = transform
        self.label_idx = label_idx
        self.root_dir = root_dir
        self.data_map = []
        self.n_data = self.data.shape[0]
        all_cat = self.data.iloc[:,label_idx]
        self.categories = list(all_cat.unique())
        self.n_category = len(self.categories)
        self.static_map = {}
        self.draw_map = {}
        for i_cat in range(self.n_category):
            my_cat = self.categories[i_cat]
            self.static_map[my_cat] = self.data.index[all_cat==my_cat].tolist()
            self.draw_map[my_cat] = self.static_map[my_cat]

        #if not self.loadMap(processed_path):
        #    print('%d/%d has been processed.' % (0,self.n_data),end='')
        #    for i in range(self.n_data):
        #        self.data_map.append([[], [], 0, 0])
        #        my_cls = self.data.iloc[i,label_idx]
        #        for data_idx in self.tmp_map[my_cls]:
        #            if data_idx==i:
        #                continue
        #            self.data_map[i][1].append(data_idx)
        #        for cls_idx in tmp_map.keys():
        #            if cls_idx==my_cls:
        #                continue
        #            else:
        #                for data_idx in tmp_map[my_cls]:
        #                    self.data_map[i][0].append(data_idx)
        #        if (i+1)%100==0:
        #            print('\r%d/%d has been processed.'%(i+1,self.n_data),end='')

        #    print('\r%d%d has been processed.' % (i+1,self.n_data), end='')
        #    print('')

    def printMe(self):
        print(self.data_map)

    def shufflePairs(self):
        for cat in self.categories:
            random.shuffle(self.static_map[cat])
        #random.shuffle(self.data_map[at_record][target_list_at])

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx,0])
        #my_f_sample_count = self.data_map[idx][2]
        #if my_f_sample_count>=len(self.data_map[idx][0]):
        #    my_f_sample_count = 0
        #    self.shufflePairs(idx,0)
        #my_t_sample_count = self.data_map[idx][3]
        #if my_t_sample_count>=len(self.data_map[idx][1]):
        #    my_t_sample_count = 0
        #    self.shufflePairs(idx,1)
        #t_img_name = os.path.join(self.root_dir, self.data.iloc[self.data_map[idx][1][my_t_sample_count], 0])
        #f_img_name = os.path.join(self.root_dir, self.data.iloc[self.data_map[idx][2][my_f_sample_count], 0])
        #self.data_map[idx][2] = my_f_sample_count + 1
        #self.data_map[idx][3] = my_t_sample_count + 1
        my_cls = self.data.iloc[idx,self.label_idx]

        draw_t_indices = random.sample(self.draw_map[my_cls], 2)
        draw_t_idx = draw_t_indices[0]
        if draw_t_idx == idx:
            draw_t_idx = draw_t_indices[1]

        draw_f_cat = random.sample(self.categories,1)
        draw_f_idx = random.sample(self.static_map[draw_f_cat[0]],1)[0]
        t_img_name = os.path.join(self.root_dir, self.data.iloc[draw_t_idx, 0])
        f_img_name = os.path.join(self.root_dir, self.data.iloc[draw_f_idx, 0])

        #img = io.imread(img_name)
        #timg = io.imread(t_img_name)
        #fimg = io.imread(f_img_name)
        img = Image.open(img_name)
        timg = Image.open(t_img_name)
        fimg = Image.open(f_img_name)

        sample = [img, timg, fimg]
        if self.transform:
            sample[0] = self.transform(sample[0])
            sample[1] = self.transform(sample[1])
            sample[2] = self.transform(sample[2])

        return sample

    def saveMap(self, output_path):
        if output_path is None:
            return False
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        t_path = os.path.join(output_path,'true_map.csv')
        f_path = os.path.join(output_path,'false_map.csv')

        with open(t_path, 'w') as output_tf:
            with open(f_path, 'w') as output_ff:
                for rec in range(self.n_data):
                    output_tf.write('%s\n' % (','.join(str(v) for v in self.data_map[rec][1])))
                    output_ff.write('%s\n' % (','.join(str(v) for v in self.data_map[rec][0])))

        return True

    def loadMap(self, map_path):
        if not map_path is None and os.path.isdir(map_path):
            #tmap_f = pd.read_csv(os.path.join(map_path,'true_map.csv'),error_bad_lines=False)
            #fmap_f = pd.read_csv(os.path.join(map_path,'false_map.csv'),error_bad_lines=False)
            tmap_f = open(os.path.join(map_path, 'true_map.csv'))
            fmap_f = open(os.path.join(map_path, 'false_map.csv'))
            print('%d/%d has been loaded.' % (0, self.n_data), end='')
            for i in range(self.n_data):
                self.data_map.append([[], [], 0, 0])
                tline = tmap_f.readline().rstrip('\n')
                fline = fmap_f.readline().rstrip('\n')
                self.data_map[i][1] = list(map(int,tline.split(',')))
                self.data_map[i][0] = list(map(int,fline.split(',')))
                if (i+1)%100==0:
                    print('\r%d/%d has been loaded.'%(i+1,self.n_data),end='')
            print('\r%d%d has been loaded.' % (i + 1, self.n_data), end='')
            print('')
            return True
        else:
            return False

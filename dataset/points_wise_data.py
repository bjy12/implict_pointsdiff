import torch 
import os
import sys
import numpy as np
import yaml
from copy import deepcopy
from torch.utils.data import Dataset
from utils import sitk_load ,sitk_save
import pdb


class RandomPoints_Dataset(Dataset):
    def __init__(self, root_path, path_dict , file_list ,config_path, n_points=10000 ,train_mode=True):
        self.root_dir = root_path
        self.train_mode = train_mode
        #pdb.set_trace()

        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path

        with open(config_path,'r') as f:
            self.config_data = yaml.safe_load(f)

            
        if self.train_mode is False:
            out_res = np.array(self.config_data['dataset']['zoom_size'])
            points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]]
            points = points.astype(np.float32)
            points = points.reshape(3, -1)
            points = points.transpose(1, 0) # N, 3
            self.points = points / (out_res - 1)
            
        else:    
            self.blocks = np.load(self._path_dict['blocks_coords'])  # [(outres * outres * outres ) , 3 ] 
            self.npoints = n_points
        
        self.name_list = file_list   

        print(" files_data len : "   , len(self.name_list))



    def __len__(self):
        return len(self.name_list)   
    
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.root_dir, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32

        return image
    
    def load_block(self, name , b_idx):
        #pdb.set_trace()
        path = self._path_dict['blocks_vals'].format(name, b_idx)
        block = np.load(path) # uint8
        return block

    def sample_points(self, points, values):
        choice = np.random.choice(len(points), size=self.npoints, replace=False)
        points = points[choice]
        values = values[choice]
        values = values.astype(np.float32) / 255.

        return points , values


    def __getitem__(self, index):
        name = self.name_list[index]
  
        #pdb.set_trace()
        if self.train_mode is False:
            points = self.points
            points_gt = self.load_ct(name)
        else:
            b_idx = np.random.randint(len(self.blocks))
            block_values = self.load_block(name, b_idx)
            block_coords = self.blocks[b_idx] # [N, 3]
            points, points_gt = self.sample_points(block_coords, block_values)
            points_gt = points_gt[None, :]
        
        ret_dict = {
            "name": name , 
            "points": points,
            "points_gt": points_gt,
        }

        return ret_dict
        


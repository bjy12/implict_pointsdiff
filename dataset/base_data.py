import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
from utils import sitk_load, sitk_save
import os
import numpy as np
from copy import deepcopy
from abc import ABC ,abstractmethod
import yaml
import pdb

# 创建一个base_data class 用来实现基本的dataset
class BaseDataset(Dataset , ABC):
    def __init__(self , root_path , file_list , _path_dict , config_path,mode,num_view):
        self.root_dir = root_path
        self.file_list = file_list 
        
        if _path_dict is not None:
            self.load_path_dict(_path_dict)
        
        self.mode = mode
        with open(config_path,'r') as f:
            self.config_data = yaml.safe_load(f)
        self.ct_h = self.config_data['projector']['nVoxel'][0]
        
        if self.mode is False:
            out_res = np.array(self.config_data['dataset']['zoom_size'])
            points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]]
            #pdb.set_trace()
            points = points.astype(np.float32)
            points = points.reshape(3, -1)
            points = points.transpose(1, 0) # N, 3
            self.points = points / (out_res - 1)
            #pdb.set_trace()
            self.points = self.points.reshape(int(out_res[0]) , int(out_res[1]) , int(out_res[2]) , 3) #  h w d 
            #pdb.set_trace()
        else:    
            self.blocks = np.load(self._path_dict['blocks_coords'])  # [(outres * outres * outres ) , 3 ] 

        self.load_path_dict(_path_dict)



    def __len__(self):
        return len(self.file_list)
    
    @abstractmethod
    def __getitem__(self, index):
        pass

    def load_path_dict(self, path_dict):
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path         
    

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
        
        #choice = np.random.choice(len(points), size=self.npoints, replace=False)
        #points = points[choice]
        #values = values[choice]
        values = values.astype(np.float32) / 255.
        # norm
        values = (values * 2 ) - 1

        return points , values
    



 
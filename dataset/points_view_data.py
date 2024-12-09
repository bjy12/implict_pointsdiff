import sys 
import numpy as np
import yaml 
import os 
import pickle
from dataset.base_data import BaseDataset
from dataset.geometry import Geometry
import pdb
import torch

class PointsAndProjectsView_Dataset(BaseDataset):
    def __init__(self, 
                 root_path:str, 
                 file_list:str, 
                 _path_dict: dict,
                 config_path: str,
                 mode: bool,  # mode  train or val   
                 num_view: int,
                 crop_type: bool,
                 ct_res:int,
                 crop_size:int,):
        super().__init__(root_path, file_list, _path_dict , config_path , mode, num_view )

        self.num_views = num_view

        self.geo = Geometry(self.config_data['projector'])
        

        self.crop_size = crop_size

        if mode is True:
            #pdb.set_trace()
            self.blocks = self.blocks.reshape(ct_res ,ct_res , ct_res ,3)


        print(" files_data len : "   , len(self.file_list))

    def sample_projections(self, name , n_view=None):
        #* view index 1 is AP 
        #* view index 0 is LV
        #* if n_view is 1 then will use AP view 
        with open(os.path.join(self.root_dir, self._path_dict['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']
            angles = data['angles'] 
            projs_max = data['projs_max']
        if n_view is None:
            n_view = self.num_views
        views = np.linspace(0,len(projs) , n_view , endpoint=False).astype(int)
        #pdb.set_trace()
        projs = projs[views].astype(float) / 255.0

        projs = projs[:,None , ...]
        angles = angles[views]
        #* not sure if this is correct
        #projs = projs * projs_max / 0.2

        return projs , angles
    
    def project_points(self,points,angles):
        points_proj = []
        for a in angles:
            p  = self.geo.project(points,a)
            #pdb.set_trace()
            #depth_c = depth_c[None, ...]
            #print(depth_c.shape)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0).astype(np.float32) # [M,N,2]

        return points_proj
    
    def view_feature(self, points ,  angles , points_proj):
        #* view feature include 
        #* 1. rot angle 2. points 3d  3. projected points_coord , 4. distance ratio , 5, direction vec
        M, N , C  = points_proj.shape
        #pdb.set_trace()
        dis_ratio_list = []
        direction_norm_list = []
        for a in angles:
            dis_ratio , direction_norm = self.geo.get_dis_plane_points_and_o(points,a)
            dis_ratio = (dis_ratio - dis_ratio.min()) / (dis_ratio.max() - dis_ratio.min() + 1e-6)
            dis_ratio_list.append(dis_ratio)
            direction_norm_list.append(direction_norm)
        dis_ratio_list = np.stack(dis_ratio_list ,axis=0)[:,:,None]
        direction_norm_list = np.stack(direction_norm_list , axis=0)
        # pdb.set_trace()
        norm_angles = (angles / (2 * np.pi)) + 0.5 
        # 
        view1_angle = np.full((1,N,1) , norm_angles[0])
        view2_angle = np.full((1,N,1) , norm_angles[1])
        view_angle = np.concatenate([view1_angle , view2_angle] , axis=0)
        
        points = np.expand_dims(points, axis=0) # 1, 4096 ,3 
        points = np.repeat(points , 2 , axis=0)
        #points = points.repeat(0,2)
        points_proj = (points_proj + 1 ) * 0.5 
        #* final view feature   points 3  view_angle 1  dis_ratio 1  direction_norm 3  points_proj 2 
        #pdb.set_trace()
        # 连接所有特征
        # 添加特征统计信息打印
        # better normlize way ? 
        view_feature = np.concatenate([
            points,             # [2,N,3]   [0,1]
            view_angle,        # [2,N,1]    [0,1]
            dis_ratio_list,    # [2,N,1]    [0,1]
            direction_norm_list,# [2,N,3]   [-1,1]
            points_proj        # [2,N,2]    [0,1]
        ], axis=2).astype(np.float32)        
        # print("\nView Feature Statistics:")
        # print(f"Points (0-2): [{view_feature[...,0:3].min():.3f}, {view_feature[...,0:3].max():.3f}]")
        # print(f"Angle (3): [{view_feature[...,3].min():.3f}, {view_feature[...,3].max():.3f}]")
        # print(f"Distance Ratio (4): [{view_feature[...,4].min():.3f}, {view_feature[...,4].max():.3f}]")
        # print(f"Direction Norm (5-7): [{view_feature[...,5:8].min():.3f}, {view_feature[...,5:8].max():.3f}]")
        # print(f"Proj Points (8-9): [{view_feature[...,8:].min():.3f}, {view_feature[...,8:].max():.3f}]")

        #* M , N , 
        return view_feature


    #def get_random_patchfy_coord_idensity(self, crop_size):
    # overwrite load_block function for specific dataset
    def load_block_all(self, name):
        path = self._path_dict['blocks_vals'].format(name, "all")
        block = np.load(path) # uint8
        return block

    def get_patchfy_coords_values(self , block_values ):
        #pdb.set_trace()
        # 128 128 128  h w d 
        block_values = block_values.reshape(self.ct_h,self.ct_h,self.ct_h,1)
        #pdb.set_trace()
        patch_coords , patch_values =  self.get_random_patchfy_coords_values(block_values)
        # concate 

        patch_coords = patch_coords.reshape(-1,3)
        patch_values = patch_values.reshape(-1,1)
        return patch_coords , patch_values


    def get_random_patchfy_coords_values(self, block_values):
        """Efficient patch sampling prioritizing non-zero regions"""
        D, H, W, _ = block_values.shape
        crop_size = self.crop_size
        #pdb.set_trace()
        # Get indices of non-zero regions
        non_zero_idx = np.nonzero(block_values[...,-1])
        if len(non_zero_idx[0]) == 0:
            # Fallback to random sampling if volume is empty
            d = np.random.randint(0, D - crop_size)
            h = np.random.randint(0, H - crop_size)
            w = np.random.randint(0, W - crop_size)
        else:
            # Random select center point from non-zero indices
            idx = np.random.randint(len(non_zero_idx[0]))
            d, h, w = [coord[idx] for coord in non_zero_idx]
            # norm cube crop 
            #pdb.set_trace()
            s_i =  np.min([d,h,w])
            s_i = np.clip(s_i - crop_size // 2 , 0 , D - crop_size)
            
            # Adjust for crop size 
            # Adjust to ensure patch fits within volume
            # d = np.clip(d - crop_size//2, 0, D - crop_size)
            # h = np.clip(h - crop_size//2, 0, H - crop_size)  
            # w = np.clip(w - crop_size//2, 0, W - crop_size)

        # patch_values = block_values[d:d+crop_size, h:h+crop_size, w:w+crop_size]
        # patch_coords = self.blocks[d:d+crop_size, h:h+crop_size, w:w+crop_size]
        
        patch_values = block_values[s_i:s_i+crop_size, s_i:s_i+crop_size, s_i:s_i+crop_size]
        #pdb.set_trace()
        if self.mode == True:
            patch_coords = self.blocks[s_i:s_i+crop_size, s_i:s_i+crop_size, s_i:s_i+crop_size]
        else:
            patch_coords = self.points[s_i:s_i+crop_size, s_i:s_i+crop_size, s_i:s_i+crop_size]
        #pdb.set_trace()
        
        return patch_coords, patch_values
    def get_local_coord(self, global_coords):
        denorm_coords = global_coords * 128
        denorm_coords = denorm_coords.astype(np.int32)
        #pdb.set_trace()
        # get denorm_coords max and min
        max_ = np.max(denorm_coords, axis=0)
        min_ = np.min(denorm_coords, axis=0)



        denorm_coords = denorm_coords - min_
        denorm_coords = denorm_coords / (max_ - min_)
        #pdb.set_trace()
        denorm_coords = denorm_coords.astype(np.float64)
        
        return denorm_coords


    def __getitem__(self, index):

        name = self.file_list[index]
        # pdb.set_trace()
        projs , angles = self.sample_projections(name)
        #pdb.set_trace()
        #projs = torch.from_numpy(np.ascontiguousarray(projs, dtype=np.float32))
        #angles = torch.from_numpy(np.ascontiguousarray(angles, dtype=np.float32))
        if self.mode is False:
            #points = self.points
            points_gt = self.load_ct(name) # h w d 
            #pdb.set_trace()
            patch_coords , patch_value = self.get_patchfy_coords_values(points_gt)
            points, points_gt = self.sample_points(patch_coords, patch_value)
            local_coord = self.get_local_coord(points)
            #pdb.set_trace()
        else:
            #pdb.set_trace()
            block_values = self.load_block_all(name)
            #pdb.set_trace()
            patch_coords , patch_value = self.get_patchfy_coords_values(block_values)
            #pdb.set_trace()
            points, points_gt = self.sample_points(patch_coords, patch_value)
            local_coord = self.get_local_coord(points)
            #pdb.set_trace()

        points_proj = self.project_points(points , angles)
        #pdb.set_trace()

        view_feature = self.view_feature(points , angles , points_proj)
        
        #pdb.set_trace()
        ret_dict = {
            #'name': name,   
            'angles': angles,  # [M , ]
            'projs': projs,    # [M , 1 , W , H]
            'points': points,  # [N , 3]
            'local_coord': local_coord, # [N , 3]
            'points_gt': points_gt, # [1, N]
            'points_proj': points_proj, # [M , N , 2]
            'view_feature': view_feature,
        }
    
        return ret_dict


        

        






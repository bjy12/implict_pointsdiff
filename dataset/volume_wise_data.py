import numpy as np
import random
import SimpleITK as sitk
from torch.utils import data
import utils
from copy import deepcopy
import yaml
import os
from utils import sitk_load , sitk_save
import pdb

class EffectiveBBoxCrop:
    def __init__(self, crop_size, threshold=0.1):
        """
        基于有效体素的bounding box裁剪
        Args:
            crop_size: 裁剪大小，例如[64,64,64]
            threshold: 判定有效体素的阈值
        """
        self.crop_size = crop_size
        self.threshold = threshold

    def compute_effective_bbox(self, image):
        """
        计算有效体素的bounding box
        Args:
            image: CT图像 [H, W, D]
        Returns:
            bbox: (min_x, max_x, min_y, max_y, min_z, max_z)
            valid: 是否找到有效的bbox
        """
        # 获取有效体素
        valid_voxels = image > self.threshold
        if not np.any(valid_voxels):
            return None, False

        # 计算bounding box
        x_indices, y_indices, z_indices = np.nonzero(valid_voxels)
        bbox = {
            'min_x': np.min(x_indices),
            'max_x': np.max(x_indices),
            'min_y': np.min(y_indices),
            'max_y': np.max(y_indices),
            'min_z': np.min(z_indices),
            'max_z': np.max(z_indices)
        }
        
        # 计算bbox的大小
        bbox['size_x'] = bbox['max_x'] - bbox['min_x'] + 1
        bbox['size_y'] = bbox['max_y'] - bbox['min_y'] + 1
        bbox['size_z'] = bbox['max_z'] - bbox['min_z'] + 1
        
        return bbox, True

    def get_valid_crop_ranges(self, bbox, image_shape):
        """
        计算有效的裁剪范围
        """
        valid_ranges = []
        
        # 确保裁剪大小不超过图像大小
        crop_size = min(self.crop_size, min(image_shape))
        
        # 计算在bbox内可以裁剪的范围
        x_start = max(0, bbox['min_x'])
        x_end = min(image_shape[0] - crop_size, bbox['max_x'])
        
        y_start = max(0, bbox['min_y'])
        y_end = min(image_shape[1] - crop_size, bbox['max_y'])
        
        z_start = max(0, bbox['min_z'])
        z_end = min(image_shape[2] - crop_size, bbox['max_z'])
        
        return {
            'x': (x_start, x_end),
            'y': (y_start, y_end),
            'z': (z_start, z_end)
        }

    def random_crop_image(self, image, points):
        """
        在有效体素区域内随机裁剪
        Args:
            image: CT图像 [H, W, D]
            points: 点云数据 [C, H, W, D]
        Returns:
            img_crop: 裁剪后的图像
            coord_crop: 裁剪后的点云数据
            success: 是否成功裁剪
        """
        # 计算有效体素的bounding box
        bbox, valid = self.compute_effective_bbox(image)
        if not valid:
            print("Warning: No valid voxels found in the image")
            return None, None, False
            
        # 获取有效的裁剪范围
        crop_ranges = self.get_valid_crop_ranges(bbox, image.shape)
        
        # 检查是否有足够的空间进行裁剪
        if (crop_ranges['x'][1] < crop_ranges['x'][0] or
            crop_ranges['y'][1] < crop_ranges['y'][0] or
            crop_ranges['z'][1] < crop_ranges['z'][0]):
            print("Warning: Not enough space for cropping")
            return None, None, False
        
        # 随机选择裁剪起点
        x0 = np.random.randint(crop_ranges['x'][0], max(crop_ranges['x'][1] + 1, crop_ranges['x'][0] + 1))
        y0 = np.random.randint(crop_ranges['y'][0], max(crop_ranges['y'][1] + 1, crop_ranges['y'][0] + 1))
        z0 = np.random.randint(crop_ranges['z'][0], max(crop_ranges['z'][1] + 1, crop_ranges['z'][0] + 1))
        
        # 执行裁剪
        img_crop = image[x0:x0+self.crop_size, 
                        y0:y0+self.crop_size, 
                        z0:z0+self.crop_size]
        
        coord_crop = points[:, x0:x0+self.crop_size, 
                             y0:y0+self.crop_size, 
                             z0:z0+self.crop_size]
        
        # 计算有效体素比例
        valid_ratio = np.mean(img_crop > self.threshold)
        
        return img_crop, coord_crop, valid_ratio > 0.1

    def check_crop_quality(self, crop):
        """
        检查裁剪质量
        """
        valid_voxels = crop > self.threshold
        valid_ratio = np.mean(valid_voxels)
        return valid_ratio > 0.1

    def crop_with_retry(self, image, points, max_attempts=10):
        """
        带重试机制的裁剪，确保裁剪质量
        """
        for attempt in range(max_attempts):
            img_crop, coord_crop, success = self.random_crop_image(image, points)
            if success:
                return img_crop, coord_crop, True
                
        print(f"Warning: Failed to find good crop after {max_attempts} attempts")
        return None, None, False




class RandomVolumeCrop(data.Dataset):
    def __init__(self, root_dir, path_dict , file_list ,config_path, crop_size ,n_points=10000, is_train=True):
        self.is_train = is_train
        self.sample_size = n_points
        self.root_dir = root_dir
        self.file_list = file_list
        self.crop_size = crop_size
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.root_dir, self._path_dict[key])
            self._path_dict[key] = path

        with open(config_path,'r') as f:
            self.config_data = yaml.safe_load(f)
        out_res = np.array(self.config_data['dataset']['zoom_size'])

        points = np.mgrid[:out_res[0], :out_res[1], :out_res[2]]
        points = points.astype(np.float32)
        #points = points.reshape(3, -1)
        #points = points.transpose(1, 0) # N, 3
        
        points = points / (out_res[0] - 1)
        self.points = points * 2 - 1

        self.cropper = EffectiveBBoxCrop(crop_size , threshold=0.0)

    
    def __len__(self):
        return len(self.file_list)
    
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.root_dir, self._path_dict['image'].format(name)),
            uint8=True
        ) # float32

        return image    
    
    def random_crop_image(self , image , points):
        # h, w, d = image.shape
        # crop_size = self.crop_size

        # # ±40 for avoiding black background region
        # x0 = np.random.randint(0, h-crop_size)
        # y0 = np.random.randint(0, w-crop_size)
        # z0 = np.random.randint(0, d-crop_size)
        # img_in = image[x0:x0+crop_size, y0:y0+crop_size, z0:z0+crop_size]
        # #pdb.set_trace()
        # coord  = points[:,x0:x0+crop_size, y0:y0+crop_size, z0:z0+crop_size]

        img_in , coord , _  = self.cropper.crop_with_retry(image , points)

        return img_in , coord
        #pdb.set_trace()


    def __getitem__(self, index):

        name = self.file_list[index]

        image = self.load_ct(name)

        patch_ , coord = self.random_crop_image(image , self.points) 

        coord = coord.reshape(3, -1)
        coord = coord.transpose(1, 0) # N, 3
        #print( coord.shape  )
        patch_ = patch_.astype(np.float32) / 255.

        #pdb.set_trace()
        if self.is_train:
            choice = np.random.choice(len(coord), size=self.sample_size, replace=False)
            coord  = coord[choice]
            gt_value = patch_.reshape(-1,1)[choice]
        else:
            gt_value = patch_.reshape(-1,1)

        return patch_ , coord , gt_value
    








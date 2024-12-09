from dataset import RandomPoints_Dataset
from dataset import RandomVolumeCrop
from dataset import PointsAndProjectsView_Dataset
from utils import get_filesname_from_txt

from torch.utils.data import DataLoader
import pdb

PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy',
    'low_res_coords': 'low_res_coords.pickle'
}



def get_dataloader(data_config):
    
    dataset_name = data_config['name']
    root_dir = data_config['base_dir']
    geo_config = data_config['geo_config_path']
    train_file_path = data_config['train_list']
    test_file_path = data_config['val_list']

    train_files_list = get_filesname_from_txt(train_file_path)
    test_files_list = get_filesname_from_txt(test_file_path)

    if dataset_name == "RandomPoints_AE": 
        n_points = data_config['n_points']
        train_dataset = RandomPoints_Dataset(root_path=root_dir,
                                            path_dict=PATH_DICT,
                                            file_list=train_files_list,
                                            config_path=geo_config,
                                            n_points=n_points,
                                            train_mode=True
                                            )
        val_dataset = RandomPoints_Dataset(root_path=root_dir,
                                        path_dict=PATH_DICT,
                                        file_list=test_files_list,
                                        config_path=geo_config,
                                        n_points=n_points,
                                        train_mode=False
                                        )
    if dataset_name == 'RandomVolume_AE':
        crop_size = data_config['crop_size']
        train_dataset = RandomVolumeCrop(root_dir=root_dir,path_dict=PATH_DICT,file_list=train_files_list,config_path=geo_config
                                          ,crop_size=crop_size ,n_points=10000,is_train=True)
        val_dataset = RandomVolumeCrop(root_dir=root_dir,path_dict=PATH_DICT,file_list=train_files_list,config_path=geo_config
                                          ,crop_size=crop_size ,n_points=10000,is_train=False)
    if dataset_name == 'PointsViewFeature':
        n_points = data_config['n_points']
        n_view = data_config['num_view']
        train_dataset = PointsAndProjectsView_Dataset(root_path=root_dir,
                                                    file_list=train_files_list,
                                                    _path_dict=PATH_DICT,
                                                    config_path=geo_config,
                                                    mode=True,  # train 
                                                    num_view=n_view,
                                                    crop_type=data_config['crop_type'],
                                                    ct_res=data_config['ct_res'],
                                                    crop_size=data_config['crop_size'],
                                                    )
        val_dataset = PointsAndProjectsView_Dataset(root_path=root_dir,
                                                    file_list=test_files_list,
                                                    _path_dict=PATH_DICT,
                                                    config_path=geo_config,
                                                    mode=False, # val 
                                                    num_view=n_view,
                                                    crop_type=data_config['crop_type'],
                                                    ct_res=data_config['ct_res'],
                                                    crop_size=data_config['crop_size'])
    #pdb.set_trace()
    #v_1 = val_dataset[0]

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset , batch_size=1 , shuffle=True )
    # else:
    #     val_loader = None
    
    

    return train_loader , val_loader
    






# main
#if __name__ == '__main__':


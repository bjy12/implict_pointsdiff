import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
import torch
import torch.nn as nn
print(torch.cuda.is_available())
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import pdb
#torch.utils.cpp_extension.clean_all()
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
#pdb.set_trace()
from model.models.diffusion_utilis.diffusion import IdensityDiffusion
from model.models.PointsViewFusion import Points_View_Fusion_ImplictDiffuse
from model.models.diffusion_utilis.ema import EMAHelper
from get_dataset import get_dataloader
from train_utils import convert_cuda
from evaluate_utils import evaluate_ssim_psnr


try:
    import lovely_tensors
    lovely_tensors.monkey_patch()
except ImportError:
    pass  # lovely tensors is not necessary but it really is lovely, I do recommend it

def create_experiment_folder(cfg):
    """
    Create a folder structure for saving experiment results based on current execution directory
    Returns the path to the experiment directory
    """
    # Get current execution directory
    #root_dir = os.path.dirname(os.path.abspath(__file__))
    #pdb.set_trace()
    current_dir = 'exp_save'
    
    # Get base results directory from config or use default
    base_dir = cfg['setting']['results_dir']
    
    # If base_dir is not absolute path, make it relative to current directory
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(current_dir, base_dir)
    
    # Get model name from config
    model_name = cfg['model']['name']
    
    # Create timestamp for unique folder
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Create experiment folder name
    exp_name = f"{model_name}_{timestamp}"
    
    # Create full path
    exp_dir = os.path.join(base_dir, exp_name)
    
    # Create necessary subdirectories
    subdirs = ['checkpoints', 'logs', 'visualizations']
    os.makedirs(exp_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    print(f"Created experiment directory at: {exp_dir}")
    return exp_dir ,subdirs


def train(config_file_path, cfg):

    torch.autograd.set_detect_anomaly(True)
    #* generate experiment local path
    #pdb.set_trace()  
    exp_dir, subdir = create_experiment_folder(cfg) # subdir ['checkpoints', 'logs', 'visualizations']
    #pdb.set_trace()

    #* setting tensorboard 
    tensorboard_logger = SummaryWriter(os.path.join(exp_dir, subdir[1]))


    #pdb.set_trace()
    #save config file to exp_file root path 
    try:
        config_filename = os.path.basename(config_file_path)
        # 创建目标文件的完整路径
        dest_path = os.path.join(exp_dir, config_filename)
        copyfile(config_file_path, dest_path)
    except:
        print('The two files are the same, no need to copy')
    print("Config file has been copied from %s to %s" % (config_file_path, exp_dir), flush=True)


    #pdb.set_trace()
    train_dl , val_dl = get_dataloader(cfg['data'])

    # model setting 
    #pdb.set_trace()
    model_cfg = cfg['model']
    #pdb.set_trace()
    model = Points_View_Fusion_ImplictDiffuse(model_cfg['image_encoder'], model_cfg['aggregation_config'] , 
                                              model_cfg['pv_encoder']  ,  model_cfg['triplane_encoder'],
                                              model_cfg['triplane_decoder'] , denoised_config=cfg['denoise_net'], 
                                              ct_res=model_cfg['ct_res'] )
    model = model.cuda()
    model.train()
    #pdb.set_trace()
    # diffusion process setting 
    idensity_diffusion  = IdensityDiffusion(cfg['standard_diffusion_config'] ,denoise_net=model)
    #pdb.set_trace()    
    # ema rate 
    ema_rate = cfg['setting']['ema_rate']
    ema_rate = list(ema_rate)
    if ema_rate is not None:
        assert isinstance(ema_rate, list)
        ema_helper_list = [EMAHelper(mu=rate) for rate in ema_rate]
        for ema_helper in ema_helper_list:
            ema_helper.register(model)
    # optimizer setting 
    #pdb.set_trace()
    optimizer = torch.optim.Adam(model.parameters() , lr = cfg['setting']['lr'])
    #pdb.set_trace()
    #* traning setting 
    time0 = time.time()
    num_epochs = cfg['setting']['epoch']
    epochs_per_ckpt = cfg['setting']['epochs_per_ckpt']
    ckpt_iter = cfg['setting']['ckpt_iter'] # for  continue training  if  not resume training  
    eval_start_epoch = cfg['setting']['eval_start_epoch']
    iters_per_logging = cfg['setting']['iters_per_logging']
    eval_per_ckpt = cfg['setting']['eval_per_ckpt']
    loader_len = len(train_dl)
    n_iters = int(loader_len * num_epochs) # number of total training steps 
    iters_per_ckpt = int(loader_len * epochs_per_ckpt) # save a ckpt every iters_per_ckpt steps
    n_iter = ckpt_iter + 1
    eval_start_iter = eval_start_epoch  *  loader_len - 1 
    # 
    num_ckpts = 0 # evey time restart training 
    log_start_time = time.time() # used to compute how much time is consumed between 2 printing log
    # n_iter from 0 to n_iters if we train the model from sratch
    while n_iter < n_iters+1:
        #model.train()
        #pdb.set_trace()
        for batch in train_dl:
            epoch_number = int((n_iter + 1) / loader_len)
            optimizer.zero_grad()
            loss_batch = idensity_diffusion.train_loss_idensity(batch)
            loss = loss_batch.mean()
            reduced_loss = loss.item()
            #pdb.set_trace()
            loss.backward()
            optimizer.step()
            # ema model update
            if ema_rate is not None:
                for ema_helper in ema_helper_list:
                    ema_helper.update(model)
            #grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #print(f"Gradient norm: {grad_norm}")
            #print loss 

            # logging each iters 
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \treduced loss: {:.6f} \tloss: {:.6f} \ttime: {:.2f}s".format(
                    n_iter, reduced_loss, loss.item(), time.time()-log_start_time), flush=True)
                log_start_time = time.time()
                tensorboard_logger.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                tensorboard_logger.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
            # save checkpoint 
            if n_iter > 0 and ( n_iter + 1) % iters_per_ckpt == 0:
                num_ckpts = num_ckpts + 1
                checkpoint_name = 'ckpt_iters_{}.pkl'.format(n_iter)
                checkpoint_states = {'iter': n_iter,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time()-time0)}
                if not ema_rate is None:
                    checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in ema_helper_list]
                torch.save(checkpoint_states, os.path.join(exp_dir,subdir[1], checkpoint_name))
                print('model at iteration %s at n_iter %d is saved' % (n_iter, epoch_number), flush=True)
            # evaluate the model at the checkpoint
            if n_iter >= eval_start_iter and num_ckpts % eval_per_ckpt==0:
                save_dir = os.path.join(exp_dir,subdir[2], 'eval_result')
                ckpt_info = '_epoch_%s_iter_%d' % (str(epoch_number).zfill(4), n_iter)
                print('\nBegin evaluting the saved checkpoint')
                evaluate_ssim_psnr(model ,idensity_diffusion , cfg ,val_dl,
                                    save_dir , tensorboard_logger , task='vis_blocks',
                                    ckpt_info=ckpt_info)

        latest_checkpoint_name = 'model_ckpt_latest.pkl'
        latest_checkpoint_states = {
            'iter': n_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_time_seconds': int(time.time()-time0),
            'epoch': epoch_number
        }
        if not ema_rate is None:
            latest_checkpoint_states['ema_state_list'] = [ema_helper.state_dict() for ema_helper in ema_helper_list]
        # 保存到 checkpoints 目录，覆盖之前的文件
        latest_save_path = os.path.join(exp_dir, subdir[0], latest_checkpoint_name)
        torch.save(latest_checkpoint_states, latest_save_path)
        print(f'Latest model saved at epoch {epoch_number}, iteration {n_iter}', flush=True)
        n_iter += 1        

            
        



#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', 
                    help='Path to configuration file')
    args = parser.parse_args()

    # 检查GPU并处理CUDA设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    #print('Visible GPUs are', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    num_gpus = torch.cuda.device_count()
    print('%d GPUs are available' % num_gpus, flush=True)

    #pdb.set_trace()
    cfg = OmegaConf.load(args.config)
    #pdb.set_trace()

    train(args.config , cfg )
    
 

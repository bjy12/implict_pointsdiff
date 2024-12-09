import numpy as np 
import os
import pdb
import time 
import pytorch_ssim_3D
import torch
from utils import sitk_save , sitk_load ,get_filesname_from_txt


def evaluate_per( net , diffusion_model , cfg ,val_dataset, tb_logger , task ):
    total_timing = []
    total_len = len(val_dataset)
    net.eval()
    total_idensity_gt = []
    total_idensity_pred = []
    total_psnr = []
    total_ssim = []
    image_size = cfg['data']['crop_size']
    shape = (cfg['setting']['num_points'] , 3+1 )
    for idx, data in enumerate(val_dataset):
        start_time = time.time()
        idensity_gt = data['points_gt'] # type np float 32 
        batch = idensity_gt.shape[0]
        #pdb.set_trace()
        idensity_gt = idensity_gt.permute(0,2,1)
        idensity_gt = idensity_gt.reshape(batch , 1 ,image_size ,image_size,image_size)
        total_idensity_gt.append(idensity_gt)
        start_time = time.time()
        if task == 'vis_blocks':
            # pred idensity 
            pred_idensity = diffusion_model.denoised_and_pred_idensity(data ,  net , shape) # pred_idensity b , n , 1 
            pred_idensity = pred_idensity.permute(0,2,1)
            pred_idensity = pred_idensity.reshape(batch,1,image_size,image_size,image_size)
        total_idensity_pred.append(pred_idensity)
        print('progress [%d/%d] %.4f, %d pred idensity' % (idx+1, total_len, idx+1/total_len, batch), flush=True)
        ssim_3d , psnr_3d = evaluate_pred_gt_ssim_psnr(total_idensity_pred , total_idensity_gt)
        total_ssim.append(np.array(psnr_3d))
        total_psnr.append(ssim_3d.cpu().numpy())
    psnr_max = np.max(total_psnr)
    psnr_mean = np.mean(total_psnr)
    psnr_min = np.min(total_psnr)
    psnr_std = np.std(total_psnr)

    ssim_min = np.min(total_ssim)
    ssim_mean = np.mean(total_ssim)
    ssim_max = np.max(total_ssim)
    ssim_std = np.std(total_ssim)
    # 打印统计结果
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("-"*50)
    print("PSNR Metrics:")
    print(f"  Max: {psnr_max:.4f}")
    print(f"  Min: {psnr_min:.4f}")
    print(f"  Mean: {psnr_mean:.4f}")
    print(f"  Std: {psnr_std:.4f}")
    print("-"*50)
    print("SSIM Metrics:")
    print(f"  Max: {ssim_max:.4f}")
    print(f"  Min: {ssim_min:.4f}")
    print(f"  Mean: {ssim_mean:.4f}")
    print(f"  Std: {ssim_std:.4f}")
    print("="*50)
    net.train()


    return total_idensity_gt , total_idensity_pred




def evaluate_pred_gt_ssim_psnr(total_idensity_pred , total_idensity_gt):
    """
    total_idensity_pred : list of pred idensity , each 
    total_idensity_gt : [ ]
    """
    #pdb.set_trace()
    with torch.no_grad():
        total_idensity_pred = torch.cat(total_idensity_pred , dim=0)
        total_idensity_gt = torch.cat(total_idensity_gt , dim=0)
        total_idensity_gt = total_idensity_gt.cuda()
        #pdb.set_trace()
        ssim_3d = pytorch_ssim_3D.ssim3D(total_idensity_gt , total_idensity_pred)
        psnr_3d = pytorch_ssim_3D.psnr3d_patch(total_idensity_gt , total_idensity_pred)

    return ssim_3d , psnr_3d 


def vis_pred_idensity_and_gt_idensity(total_idensity_gt , total_idensity_pred ,image_size ,  save_dir , ckpt_info ,random_vis=5):
    
    total_idensity_pred = torch.cat(total_idensity_pred , dim=0)
    total_idensity_gt = torch.cat(total_idensity_gt , dim=0)
    #pdb.set_trace()
    N_case = total_idensity_gt.shape[0]
    #pdb.set_trace()
    if random_vis is not None:
        # 确保random_vis不超过总样本数
        random_vis = min(random_vis, N_case)
        # 随机选择样本索引
        random_indices = torch.randperm(N_case)[:random_vis]
        selected_gt = total_idensity_gt[random_indices]
        selected_pred = total_idensity_pred[random_indices]

    for i in range(selected_gt.shape[0]):
        pred_vis_path = os.path.join(save_dir,'%d_pred_%s.nii.gz' % (i , ckpt_info))
        gt_vis_path = os.path.join(save_dir, '%d_gt_%s.nii.gz' % (i , ckpt_info))
        pred_ = selected_pred[i].reshape(1,image_size,image_size,image_size).squeeze().cpu().numpy()
        gt_ = selected_gt[i].reshape(1,image_size,image_size,image_size).squeeze().cpu().numpy()
        sitk_save(pred_vis_path ,pred_ )
        sitk_save(gt_vis_path , gt_)

def evaluate_ssim_psnr( net , diffusion_model , cfg , val_dataset ,save_dir , tb_logger,task ,ckpt_info):

    """
    net : condition and denoised network 
    diffusion_model : denoised process 
    cfg: get val dataloader 
    val_dataset:  验证数据集
    save_dir: 可视化保存路径 
    tb_logger: 保存数值到tensorboard 
    task:  选择是否可视化
    """
    os.makedirs(save_dir , exist_ok=True)

    total_idensity_gt , total_idensity_pred =  evaluate_per(net , diffusion_model,cfg, val_dataset, tb_logger , task)


    vis_pred_idensity_and_gt_idensity(total_idensity_gt , total_idensity_pred , image_size=cfg['data']['crop_size'] , 
                                      save_dir=save_dir ,random_vis=5 ,ckpt_info=ckpt_info)


if __name__ == '__main__':
    pred_list = []
    gt_list = []
    for i in range(0,10):
        pred_ = torch.randn(1,16,16,16)
        gt_ = torch.randn(1,16,16,16)
        pred_list.append(pred_)
        gt_list.append(gt_)
    ssim , psnr =  evaluate_pred_gt(pred_list , gt_list)





import torch 
import glob 
import os
import sys
import pdb
from torch.utils.tensorboard import SummaryWriter


def get_latest_checkpoint(ckpt_dir):
    """获取最新的检查点文件"""
    checkpoints = glob.glob(os.path.join(ckpt_dir, 'model_param_*.pkl'))
    if not checkpoints:
        return None
    
    # 从文件名中提取epoch数并找到最新的
    latest_epoch = max([int(ckpt.split('_')[-1].split('.')[0]) for ckpt in checkpoints])
    return os.path.join(ckpt_dir, f'model_param_{latest_epoch}.pkl'), latest_epoch



def save_checkpoint(model, optimizer, epoch, ckpt_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, ckpt_path)



def load_checkpoint(model, optimizer, ckpt_path):
    """加载完整检查点进行训练恢复"""
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] , model , optimizer

def get_tensorboard_writer(log_dir, resume_tf=None):
    """
    获取TensorBoard writer，如果是恢复训练则查找已有的日志
    """
    if resume_tf and os.path.exists(log_dir):
        # 查找最后一个事件文件
        event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
        if event_files:
            print(f"Found existing TensorBoard logs in {log_dir}")
            # 使用已存在的日志目录，TensorBoard会自动追加
            return SummaryWriter(log_dir)
    
    # 新建日志或目录不存在的情况
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item
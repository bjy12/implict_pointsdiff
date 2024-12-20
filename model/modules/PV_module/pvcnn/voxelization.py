import torch
import torch.nn as nn
import pdb
import model.modules.PV_module.pvcnn.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        #pdb.set_trace()
        """
        coord : b n 3  3(x y z)
        """
        #pdb.set_trace()
        coords = coords.detach()
        # norm_coords = coords - coords.mean(2, keepdim=True)
        # if self.normalize:
        #     pdb.set_trace()
        #     norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        # else:
        #     norm_coords = (norm_coords + 1) / 2.0
        #pdb.set_trace()
        norm_coords = torch.clamp(coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')

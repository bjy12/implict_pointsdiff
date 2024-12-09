import torch
import torch.nn as nn
import pdb

from model.models.denoised.pvcnn_transformer import PVCNN_Attention_Base
from torch import Tensor


class DenoisedModelBlock(nn.Module):
    def __init__(self,
                 model_type: str = 'pvcnn_att',
                 in_channels: int = 3,
                 out_channels: int = 1,
                 embed_dim: int = 64 ,
                 use_att: bool = True,
                 dropout: int = 0.1,
                 width_multiplier: int = 1,
                 voxel_resolution_multiplier: int = 1, 
                 sa_blocks: list = None,
                 fp_blocks: list = None

    ):
        super().__init__()

        self.model_type = model_type

        if self.model_type == 'pvcnn_att':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = PVCNN_Attention_Base(
                embed_dim=embed_dim,
                num_classes=out_channels,
                extra_feature_channels=(in_channels - 3),
                dropout=dropout, width_multiplier=width_multiplier, 
                voxel_resolution_multiplier=voxel_resolution_multiplier,
                sa_blocks = sa_blocks,
                fp_blocks = fp_blocks,
                use_att=use_att   
            )
            self.model.classifier[-1].bias.data.normal_(0,1e-6)
            self.model.classifier[-1].bias.data.normal_(0,1e-6)
        else:
            raise NotImplementedError()
        
    def forward(self, inputs: Tensor , t: Tensor , local_coords:Tensor = None,global_coord: Tensor =None ):
        #pdb.set_trace()
        
        if self.model_type == 'pvcnn_att':
            with self.autocast_context:
                out_put = self.model( inputs , t , local_coords ,global_coord)

        return out_put




    

            
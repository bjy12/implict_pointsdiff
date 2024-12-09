import torch 
import torch.nn as nn
import pdb 
import numpy as np 
from timm.models.vision_transformer import Attention, LayerScale, DropPath, Mlp
from torch import Tensor
from typing import Optional
from model.models.denoised.denoisedmodel import DenoisedModelBlock



class PointCloudModelBlock(nn.Module):

    def __init__(
        self, 
        *, 
        # Point cloud model
        dim: int,
        model_type: str = 'pvcnn',
        dropout: float = 0.1,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
        sa_blocks : list = None ,
        fp_blocks : list = None ,
        # Transformer model
        num_heads=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_attn=False
    ):
        super().__init__()

        # Point cloud model
        self.norm0 = norm_layer(dim)
        self.point_cloud_model = DenoisedModelBlock(model_type=model_type, 
            in_channels=dim, out_channels=dim, embed_dim=dim, dropout=dropout, 
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier ,
            sa_blocks = sa_blocks , fp_blocks = fp_blocks)
        self.ls0 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention
        self.use_attn = use_attn
        if self.use_attn:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def apply_point_cloud_model(self, x: Tensor, local_coord: Optional[Tensor] = None , global_coord: Optional[Tensor] = None ,t: Optional[Tensor] = None ) -> Tensor:
        t = t if t is not None else torch.zeros(len(x), device=x.device, dtype=torch.long)
        return self.point_cloud_model(x, t , local_coord , global_coord)

    def forward(self, x: Tensor , t: Optional [Tensor] = None  , global_coord : Optional [Tensor] = None , local_coord : Optional [Tensor] = None):
   
        x = x + self.drop_path0(self.ls0(self.apply_point_cloud_model(self.norm0(x) , local_coord , global_coord , t)))
        
        #pdb.set_trace()

        if self.use_attn:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PointCloudTransformerModel(nn.Module):
    def __init__(self, num_layers: int, in_channels: int = 3, out_channels: int = 3, embed_dim: int = 64 , norm_dim: int = 256, **kwargs):
        super().__init__()
        self.num_layers = num_layers # number of point cloud model blocks 
        self.input_projection = nn.Linear(in_channels, embed_dim)
        #self.blocks = nn.Sequential(*[PointCloudModelBlock(dim=embed_dim, **kwargs) for i in range(self.num_layers)])
        self.blocks = nn.ModuleList([PointCloudModelBlock(dim=embed_dim, **kwargs) for i in range(self.num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, out_channels)

    def forward(self, inputs: Tensor , t: Tensor , global_coords: Tensor = None,local_coords: Tensor = None , condition: Tensor = None) -> Tensor:

        """ inputs [b , n, c1] c1 ,v
            t [b,time_steps]
            gloabl_coord [b , n, 3]  x y z  for position learning 
            local_coords [b , n, 3]  x y z  for voxelization
            condition [b, n, c2] 
        """
        #pdb.set_trace()
        features = torch.cat([inputs , condition] , dim=2) # noised idensity and condition features 
        # process features 
        x = self.input_projection(features)
        # inputs features 
        for block in self.blocks:
            x = block(x , t , global_coords , local_coords)
        x = self.output_projection(x)
        return x

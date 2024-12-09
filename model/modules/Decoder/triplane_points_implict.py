import torch 
import torch.nn as nn
import os
import pdb
from torch_scatter import scatter_mean , scatter_max
from model.modules.Encoder.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from model.modules.Decoder.layers import ResnetBlockFC

import torch.nn.functional as F

class Multi_Scale_Triplane_Encoder(nn.Module):
    def __init__(self, res_plane , plane_feat_dim , padding , plane_type):
        super(Multi_Scale_Triplane_Encoder, self).__init__()

        self.res_plane = res_plane
        self.plane_feat_dim = plane_feat_dim 
        self.padding = padding
        self.plane_type = plane_type
        self.fusion_dim = 256
        self.triplane_feature_generator = nn.ModuleList([ 
            triplane_feature_generator(self.res_plane[i] , self.plane_feat_dim[i] , 
                                       self.padding , self.plane_type) for i in range(3)])
        
        # self.triplane_fusion_ = EfficientTriplaneFusion(plane_feat_dims = self.plane_feat_dim ,
        #                                                  fusion_dim = self.fusion_dim )

    def forward(self , pv_fusion_f, global_coord_stack):
        trip_feats = []
        for pv_fusion_ , coord_  , tripane_generator in zip(pv_fusion_f , global_coord_stack ,self.triplane_feature_generator):
                plane_feature = tripane_generator(pv_fusion_ , coord_)
                trip_feats.append(plane_feature)
        #pdb.set_trace()

        #fuse_multi_scale_trip_feats =  self.triplane_fusion_(trip_feats)

        return trip_feats


class triplane_feature_generator(nn.Module):
    def __init__(self , res , feat_dim , padding , plane_type):
        super(triplane_feature_generator, self).__init__()

        self.res_plane = res 
        self.c_dim_plane = feat_dim
        self.padding = padding
        self.plane_type = plane_type

    def generate_plane_features(self, coord  , p_f , plane='xz'):
        #pdb.set_trace()
        xy = normalize_coordinate(coord.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy , self.res_plane)
        #pdb.set_trace()
        fea_plane = p_f.new_zeros(p_f.shape[0], self.c_dim_plane, self.res_plane**2)

        p_f = p_f.permute(0,2,1)
        # pdb.set_trace()
        fea_plane = scatter_mean(p_f , index , out = fea_plane)

        fea_plane = fea_plane.reshape(coord.size(0) , self.c_dim_plane , self.res_plane , self.res_plane)

        return fea_plane
                                                  

    def forward(self, p_feats , coord):
        #pdb.set_trace()
        feat_triplane = {}
        if 'xz' in self.plane_type:
            feat_triplane['xz'] = self.generate_plane_features(coord, p_feats, plane='xz')
        if 'xy' in self.plane_type:
            feat_triplane['xy'] = self.generate_plane_features(coord, p_feats, plane='xy')
        if 'yz' in self.plane_type:
            feat_triplane['yz'] = self.generate_plane_features(coord, p_feats, plane='yz')
        #pdb.set_trace()
        return feat_triplane
    


class EfficientTriplaneFusion(nn.Module):
    def __init__(self, plane_feat_dims , fusion_dim):
        super(EfficientTriplaneFusion, self).__init__()

        self.n_scales = len(plane_feat_dims)

        self.feat_transforms = nn.ModuleList([
            nn.Conv2d(dim, fusion_dim, 1) 
            for dim in plane_feat_dims
        ])

        # 特征融合注意力
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_dim, fusion_dim//4, 1),
            nn.ReLU(True),
            nn.Conv2d(fusion_dim//4, self.n_scales, 1),
            nn.Softmax(dim=1)
        )        
        # 每个平面的特征增强模块
        self.enhance_module = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(True),
            # 第二个卷积块
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(True),
            # 残差连接的1x1卷积
            nn.Conv2d(fusion_dim, fusion_dim, 1, bias=False),
            nn.BatchNorm2d(fusion_dim)
        )
        
        # SE注意力模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_dim, fusion_dim//16, 1),
            nn.ReLU(True),
            nn.Conv2d(fusion_dim//16, fusion_dim, 1),
            nn.Sigmoid()
        )


    def enhance_features(self, x):
        """特征增强函数"""
        identity = x
        out = self.enhance_module(x)
        # SE注意力
        se_weight = self.se(out)
        out = out * se_weight
        # 残差连接
        out = out + identity
        return out
    

    def fuse_multi_scale(self, scale_feats):
        """融合单个平面的多尺度特征"""
        # 获取最高分辨率
        #pdb.set_trace()
        max_resolution = scale_feats[0].shape[-2:]
        
        # 特征变换和上采样对齐
        aligned_feats = []
        for feat, transform in zip(scale_feats, self.feat_transforms):
            feat = transform(feat)
            if feat.shape[-2:] != max_resolution:
                feat = F.interpolate(feat, size=max_resolution, 
                                mode='bilinear', align_corners=True)
            aligned_feats.append(feat)
        #pdb.set_trace()
        # 堆叠特征
        stacked_feats = torch.stack(aligned_feats, dim=1)  # [B, S, C, H, W]
        
        # 计算注意力权重
        base_feat = aligned_feats[0]
        attention_weights = self.scale_attention(base_feat)  # [B, S, 1, 1]
        
        # 加权融合
        fused = (stacked_feats * attention_weights.unsqueeze(2)).sum(dim=1)
        
        enhanced = self.enhance_features(fused)

        return enhanced
        
    def forward(self , trip_feats):
        """
        input trip_feats   dict [xy yz xz] 
        """
        #pdb.set_trace()
        # 提取每个平面的多尺度特征
        xy_scales = [f['xy'] for f in trip_feats]
        yz_scales = [f['yz'] for f in trip_feats]
        xz_scales = [f['xz'] for f in trip_feats]
        
        
        multi_scale_fuse_feat = {
            'xy': self.fuse_multi_scale(xy_scales),
            'yz': self.fuse_multi_scale(yz_scales),
            'xz': self.fuse_multi_scale(xz_scales)
        }



        #pdb.set_trace()
        
        return multi_scale_fuse_feat
    


class TriplaneImplictDecoder(nn.Module):
    def __init__(self , plane_feat_dim = 256 , hidden_size = 512  , padding = 0.1 , sample_mode = 'bilinear' , n_blocks=2):
        super(TriplaneImplictDecoder, self).__init__()

        self.plane_feat_dim = plane_feat_dim
        self.global_coord_embed = nn.Linear(3,hidden_size)
        self.padding = padding
        self.sample_mode = sample_mode
        self.n_blocks = n_blocks
        in_dim = 1792
        #pdb.set_trace()
        self.mlp_fuse_multi_scale_linear = nn.Sequential(
            nn.Linear( in_dim ,  in_dim // 4),
            nn.LayerNorm( in_dim // 4 ) , 
            nn.ReLU(),
            nn.Linear(in_dim//4 , self.plane_feat_dim)
        )

        self.fc_c = nn.ModuleList([
                nn.Linear(self.plane_feat_dim, hidden_size) for i in range(n_blocks)
            ])
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(self.n_blocks)
        ])


    
    def sample_plane_feature(self, p, c, plane='xz'):
        #pdb.set_trace()
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        
        return c 

    def forward(self, global_coord  ,trip_plane_f_list):

        """
        input:
        global_coord : b n 3 (x y z)   position in all ct grid 
        trip_plane_f : dict{ xy  yz zy }  b c h w   - local_feature  
        local_coord : b n 3 (x y z)  position in crop patch 
        """
        #pdb.set_trace()
        #pdb.set_trace()
                
        position = global_coord[0]

        coord_for_embedding = position.clone()
        # plane_type = list(trip_plane_f_list[0].keys())            
        local_f = []
        for trip_plane_f in trip_plane_f_list:
            # sample local points tri_plane_f  
            p_xz = self.sample_plane_feature(position , trip_plane_f['xz'] , plane = 'xz')
            p_xy = self.sample_plane_feature(position , trip_plane_f['xy'] , plane = 'xy')
            p_yz = self.sample_plane_feature(position , trip_plane_f['yz'] , plane = 'yz')
            # fusion local_feature  use add all p_f   
            local_f.append(p_xz + p_xy + p_yz)
            #pdb.set_trace()
        #pdb.set_trace()    
        local_f = torch.concatenate(local_f,dim=1)
        local_f = local_f.permute(0,2,1)
        local_f = self.mlp_fuse_multi_scale_linear(local_f)
        #pdb.set_trace()
        # embedding global points postion
        p_global =  self.global_coord_embed(coord_for_embedding)
        # pdb.set_trace()
        for i in range(self.n_blocks):
            p_global = p_global + self.fc_c[i](local_f)

            p_global = self.blocks[i](p_global)

        #pdb.set_trace()
        loacal_condition  = p_global # b n 128
        return loacal_condition






import pdb 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.PV_module.pvcnn.voxelization import Voxelization
from model.modules.PV_module.pvcnn.shared_mlp import SharedMLP as SharedMLP_Voxel
from model.modules.PV_module.pvcnn.functional import trilinear_devoxelize
#from model.modules.Encoder.fusion_wise.fusion_base_modules import MLP
from model.modules.PV_module.struct_feat_transformer.self_struct_transformer import RPETransformerLayer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn import MLP
from torch import LongTensor, Tensor
from torch_geometric.utils import softmax
from einops import rearrange


# Default activation, BatchNorm, and resulting MLP used by RandLA-Net authors
lrelu02_kwargs = {"negative_slope": 0.2}
bn099_kwargs = {"momentum": 0.01, "eps": 1e-6}

class SharedMLP(MLP):
    """SharedMLP following RandLA-Net paper."""

    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs["plain_last"] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs["act"] = kwargs.get("act", "LeakyReLU")
        kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
        super().__init__(*args, **kwargs)
def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

class MultiScale_Points_Volume_Encoder(nn.Module):
    def __init__(self, in_channels , out_channels,patch_res , scale_factor, k, hidden_dim,decimation=4) :
        super(MultiScale_Points_Volume_Encoder , self).__init__()
        #pdb.set_trace()
        self.in_channels = in_channels
        self.out_channels = out_channels
        eps = 1e-4
        self.patch_res = patch_res
        self.scale_factor = scale_factor

        self.decimation = decimation

        self.voxel_encoders = nn.ModuleList([
            MultiScaleVoxelGatingFusion(in_channels=in_ch , out_channels=out_ch , 
                                        eps=eps ,patch_res=patch_res , 
                                        scale_factor=scale_factor)
            for in_ch ,out_ch in zip(self.in_channels, self.out_channels)
        ])

        self.points_encoders = nn.ModuleList([
            DilatedResidualBlock_RanLA(d_in=in_ch, d_out=out_ch, num_neighbors=k)
            for in_ch, out_ch in zip(self.in_channels, self.out_channels)
        ]) 
        self.gsa_channels =  [out_cha + 128 for out_cha in self.out_channels]
        #pdb.set_trace()
        self.hidden_dim = hidden_dim
        self.gsa_module = nn.ModuleList([
            RPETransformerLayer(d_model = geo_ch , num_heads = 4 )
            for geo_ch  in self.out_channels
        ])
        #pdb.set_trace()
    def forward(self, points_f , global_coord , local_coord ):

        """
        points_f : 点的特征 b n c   
        local_coord : patch坐标系下的位置  b n c  for voxel encoder create multi scale volume feature  
        global_coord: 整个ct坐标系下的位置 b n c  for points encoder  get position relative  feature 
        return:

        """
        #pdb.set_trace()
        #* input process 
        b ,n , c  = points_f.shape
        d = self.decimation
        points_f = points_f.contiguous()
        decimation_ratio = 1

        # 
        points_branch_feat_stack = []
        voxel_branch_feat_stack = []
        local_coords_stack = []
        global_coords_stack = []
        geo_features_stack = []

        current_points_f = points_f 
        current_local_coord = local_coord
        current_global_coord = global_coord

        for i , (voxel_encoder , points_encoder ) in enumerate(zip(self.voxel_encoders , self.points_encoders)):
            current_n = n // decimation_ratio 

            # get current scale points features and volume features 
            # pdb.set_trace()
            points_branch_f , pos  ,rec_features = points_encoder(current_global_coord , current_points_f)
            voxel_branch_f = voxel_encoder(current_local_coord , current_points_f)

            # save features and coords 
            points_branch_feat_stack.append(points_branch_f)
            voxel_branch_feat_stack.append(voxel_branch_f)
            geo_features_stack.append(rec_features)
            local_coords_stack.append(current_local_coord)
            global_coords_stack.append(current_global_coord)
            # prepare next scale input             
            if i < len(self.voxel_encoders) - 1 :
                decimation_ratio *= d 
                idx = torch.randperm(current_n)[:current_n//d]
                current_points_f = points_branch_f[:,idx,:]
                current_local_coord = current_local_coord[:,idx,:]
                current_global_coord = current_global_coord[:,idx,:]
        #pdb.set_trace()
        encoder_out = []
        for geo_f , points_f , voxel_f , gsa_module in zip(geo_features_stack,
                                                           points_branch_feat_stack ,
                                                           voxel_branch_feat_stack,
                                                           self.gsa_module):
            #pdb.set_trace()
            voxel_f = voxel_f.permute(0,2,1)
            f_p_v = torch.cat([voxel_f , points_f] , dim=2)
            final_f , attention_scores = gsa_module(f_p_v ,f_p_v , geo_f)
            encoder_out.append(final_f)

        #pdb.set_trace()

        return encoder_out , local_coords_stack , global_coords_stack

        
class MultiScaleVoxelGatingFusion(nn.Module):
    def __init__(self , in_channels , out_channels , eps ,patch_res , scale_factor):
        super(MultiScaleVoxelGatingFusion , self).__init__()
        
        self.patch_res = patch_res
        self.voxel_scale = []
        #pdb.set_trace()
        for factor in scale_factor:
            self.voxel_scale.append(patch_res//factor)
        #pdb.set_trace()
        self.finest_size =  self.voxel_scale[0]
        # 
        self.voxeliazers = nn.ModuleList([])
        for voxel_res in self.voxel_scale:
            voxelizer = Voxelization(voxel_res, normalize=True)  # 创建实例
            self.voxeliazers.append(voxelizer)  # 添加实例到 ModuleList


        # 1. 多尺度体素卷积
        self.voxel_convs = nn.ModuleList([
            nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding= 1),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            ) for _ in self.voxel_scale
        ])        


        self.gate_net = Voxel_Gating_Fusion(out_channels, scale_factor)

    def forward(self,local_coord, p_features ):
        #pdb.set_trace()
        """
        input:
        local_coord : b , n , c (xyz) 
        p_features : b  , n , c 
        return :  
        devoxel_feature : b , out_channels , n 
        """
        multi_scale_features = [] # last scale is finest feature
        p_features = p_features.permute(0,2,1)
        local_norm_coord = None
        for i , (conv , voxelizers) in enumerate(zip(self.voxel_convs , self.voxeliazers)):
            #pdb.set_trace()
            if i == 0 :
                #pdb.set_trace()
                current_feature , voxel_coords = voxelizers(p_features , local_coord )
                local_norm_coord = voxel_coords

            #current_feature = F.avg_pool3d(v_features, kernel_size=target_size)
            #pdb.set_trace()
                current_feature = conv(current_feature)
            else:

                current_feature , voxel_coords = voxelizers(p_features , local_coord )
                current_feature = conv(current_feature)
            multi_scale_features.append(current_feature)
        #pdb.set_trace()
       
        upsample_features = []
        #pdb.set_trace()
        for i , down_sample_f in enumerate(multi_scale_features):
            if i == 0:
                upsample_f = down_sample_f
            else:
                #pdb.set_trace()
                upsample_f = F.interpolate(
                    down_sample_f,
                    size=self.finest_size,
                    mode='trilinear',
                    align_corners=True,
                )
            upsample_features.append(upsample_f)
        
        #pdb.set_trace()
        # fuse feature by gate net 
        fused_voxel_features = self.gate_net(upsample_features)
        #pdb.set_trace()
        voxel_res = fused_voxel_features.shape[2]
        #pdb.set_trace()
        local_norm_coord = local_norm_coord.permute(0,2,1)
        #pdb.set_trace()
        devoxel_feature = trilinear_devoxelize(fused_voxel_features, local_norm_coord, 
                                               voxel_res,True)
        #pdb.set_trace()
        return devoxel_feature



class Voxel_Gating_Fusion(nn.Module):
    def __init__(self , feature_dim , voxel_scale):
        super(Voxel_Gating_Fusion , self).__init__()
        # 3 scale 
        self.mlp_layer = nn.ModuleList([
            SharedMLP_Voxel(feature_dim , 3) for _ in range(len(voxel_scale))
        ])

        self.mlp_fuse_layer = SharedMLP_Voxel(feature_dim,feature_dim)

    def forward(self, multi_scale_features):
        #pdb.set_trace()
        B,C,H,W,D = multi_scale_features[0].shape
        gates = []
        scales_f = []
        for i , features in enumerate(multi_scale_features):
            #pdb.set_trace()
            features = features.permute(0,2,3,4,1) # b h w d c 
            #pdb.set_trace()
            features = rearrange(features,"b h w d c -> b c (h w d) " , c=C , b = B ,h=H ,w=W ,d=D)
            scales_f.append(features)
            gate = self.mlp_layer[i](features)
            gate = torch.sigmoid(gate)  # b h w d c 
            gates.append(gate)

        #pdb.set_trace()
        # 计算门控权重的和 (公式2: ∑Gi)
        gates_stack = torch.stack(gates, dim=0)
        gates_sum = torch.sum(gates_stack , dim=0)

        # 应用softmax获取归一化权重
        gates_softmax = F.softmax(gates_sum, dim=1)  # 在特征维度上进行softmax # b m_scale n_dim

        gates_softmax = gates_softmax.permute(0,2,1)

        scale_0_weight = gates_softmax[:,:,0].unsqueeze(1) # b 1 512 
        scale_1_weight = gates_softmax[:,:,1].unsqueeze(1) # b 1 512 
        scale_2_weight = gates_softmax[:,:,2].unsqueeze(1) # b 1 512 
        #pdb.set_trace()
        fuse = scale_0_weight * scales_f[0] + scale_1_weight * scales_f[1] + scale_2_weight*scales_f[2]
        fuse_f = self.mlp_fuse_layer(fuse)
        
        fuse_f = rearrange(fuse_f,'b c (h w d ) -> b c h w d ' ,  c=C , b = B ,h=H ,w=W ,d=D)
        
        return fuse_f


# class LPFE(nn.Module):
#     def __init__(self , in_channels , out_channels , k=16):
#         super(LPFE, self).__init__()
#         self.k = k
#         # self.geo_mlp = nn.Sequential(
#         #     nn.Linear(7,128)  # 3(pi) + 3(pki) + 3(disi)
#         # )
#         self.geo_mlp = SharedMLP(in_channels=7,out_channels=128)
#         self.mlp_attention = MLP(channels=[3*out_channels , out_channels])

#         # graph cnn
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv1 = nn.Sequential(nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(4*in_channels, out_channels, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.shortcut = MLP([in_channels, out_channels])
#         self.mlp = MLP([in_channels , out_channels] )
#         self.mlp2 = MLP([2*out_channels , out_channels])

#         self.lrelu = nn.LeakyReLU()

#     def get_geometric_features(self, points, knn_idx):
#         """
#         从RandLA_Net Local中学习
#         Args:
#             points: (B, N, 3) 点坐标
#             knn_idx: (B, N, k) K近邻索引
#         Returns:
#             geo_features: (B, N, C) 每个点pi的几何特征

#             RandLA_Net 
#         """
#         pdb.set_trace()
#         B, N, _ = points.shape
#         _, _, k = knn_idx.shape
        
#         # 1. 获取中心点坐标 pi
#         center_points = points  # [B, N, 3]
#         # 2. 获取近邻点坐标 pik
#         points_expanded = points.unsqueeze(2) # [b , n , 1 ,3]
#         knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, N, k, 3]
#         neighbor_points = torch.gather(points_expanded.expand(-1, -1, k, -1), 1, knn_idx_expanded)  # [B, N, k, 3]
#         # 3. 计算相对距离 disi (公式3)
#         dist = torch.norm(neighbor_points - center_points.unsqueeze(2), p=2, dim=-1)  # [B, N, k]

#         # 4. 特征拼接 (公式4: fig = mlp(pi ⊕ pik ⊕ disi))
        
#         geo_features = torch.cat([
#             center_points.unsqueeze(2).repeat(1, 1, k, 1),  # pi: [B, N, k, 3]
#             neighbor_points,                                    # pik: [B, N, k, 3]
#             dist.unsqueeze(-1)                                   # disi: [B, N, k, 1]
#         ], dim=-1)  # [B, N, k, 7]
#         pdb.set_trace()

#         geo_features = rearrange(geo_features , 'b n k c ->  b (n k) c ' ,b=B , n=N , k=k ) # b n*k c
#         geo_features = geo_features.permute(0,2,1)


#         # 5. MLP处理 (公式4的mlp部分)  
#         geo_features = self.geo_mlp(geo_features)  # [B ,C , n*k]
#         geo_features = geo_features.view(B, N, k, -1)    # [B, N, k, C]
#         geo_features = geo_features.permute(0,2,1,3) # B k N C 

#         return geo_features 
        
#     def get_graph_feature(self, x , k , knn_idx):
#         #pdb.set_trace()
#         batch_size = x.size(0)
#         num_points = x.size(2)
#         x = x.view(batch_size, -1, num_points) 
#         if knn_idx is None:
#             knn_idx = knn(x, k=k)                       # (batch_size, num_points, k)
#         _, num_dims, _ = x.size()
#         x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
#         feature = x.view(batch_size*num_points, -1)[knn_idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)

#         x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

#         feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

#         return feature 

#     def get_dgcnn_features(self, points_f,knn_idx):
#         "points_f : b c n "
#         #pdb.set_trace()
#         points_f = points_f.permute(0,2,1)
#         points_f = points_f.contiguous()
#         b , _ ,n = points_f.size()
#         #pdb.set_trace()
#         f = self.get_graph_feature(points_f , self.k , knn_idx) # b in_channels n -> b 2*in_channels n k 
#         f = self.conv1(f)  # b 2*in_channels n k -> b  2*in_channels n k
#         f1 = f.max(dim=-1, keepdim=False)[0]  # b 2*in_channels n 

#         f = self.get_graph_feature(f1 , self.k , knn_idx) # b 2*in_channels n -> b 4*in_channels n k 
#         f = self.conv2(f) # b 4*in_channels n k -> b 2*inchannels n k

#         return f
    
#     def forward(self,coord, points_f ):
#         """
#         input:
#         points_f : 点的特征 b n in_channels  
#         coord : 全局点的坐标 b n c (xyz)
#         return : 
#         points_finest_f : b out_channels n 

#         """
#         #残差连接 
#         b , n , p_c = points_f.shape
#         short_cut_f = self.shortcut(points_f.view(-1,p_c)) # b*n c_out  4096 256 

#         # 计算k邻近点

#         pdb.set_trace()
#         dist = torch.cdist(coord , coord)
#         _, knn_idx = torch.topk(dist, self.k+1, dim=2, largest=False)
#         knn_idx = knn_idx[:, :, 1:]  # Remove sel
#         # 计算相关位置特征
#         #pdb.set_trace()
#         geometric_features = self.get_geometric_features(coord , knn_idx) # b k n 128  
#         # 计算图卷积特征
#         #pdb.set_trace()
#         dgcnn_features = self.get_dgcnn_features(points_f , knn_idx) # b 2c n k 
#         #pdb.set_trace()
#         local_feature = torch.cat([
#                 geometric_features.permute(0,3, 2, 1),
#                 dgcnn_features
#             ], dim=1)  # [B, (128+2c), n, k]
#         #pdb.set_trace()
#         # clone local_feature get fp for  transformer   
#         geo_features = local_feature.clone()  
#         # attentive pool 
#         local_feature = local_feature.permute(0,2,3,1)
#         b , n  , k  , c_local  = local_feature.shape
#         local_feature =  local_feature.view(-1,c_local)
#         att_features  =  self.mlp_attention(local_feature) # b*n*k d_out 
#         att_scores =  F.softmax(att_features , dim=-1)
#         poolinged_feature =  att_scores * local_feature # b*n d_out
#         # 
#         poolinged_feature = self.mlp2(poolinged_feature)

#         points_f = self.lrelu(poolinged_feature+short_cut_f) # b*n d_out 

#         points_f = rearrange(points_f , " (b n) d_out -> b n d_out")





#         return points_f , geo_features



class DilatedResidualBlock_RanLA(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 num_neighbors,
    ):
        super().__init__()
        self.k = num_neighbors
        self.d_in = d_in
        self.d_out = d_out
       
        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8]) # 128  -> 32 
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None) # 32 
        # return fusion feature and  attentioned fusion feature 
        # attentioned fusion feature is f_p for transformer concate f_v 
        # other fusion feauture is rec  

        self.lfa1 = LocalFeatureAggregation(d_out // 4 , return_mode='both') # 256 / 4 ->  64
        self.lfa2 = LocalFeatureAggregation(d_out // 2 , return_mode='single')
        
        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)      


    def forward(self,pos,x ):  
        B , N , C = x.shape
        x = x.view(B*N , -1)
        pos = pos.view(B*N , -1)
        #pdb.set_trace()
        batch = torch.arange(B , device=x.device ).repeat_interleave(N)

        edge_index = knn_graph(pos , k=self.k ,batch= batch , loop=True)
        #pdb.set_trace()
        short_cut_x = self.shortcut(x)
        x = self.mlp1(x) # 32 
        #pdb.set_trace()
        x , rec = self.lfa1(edge_index , x , pos) # x b*n d_out//4  , rec  b*n d_out
        #pdb.set_trace()
        x  = self.lfa2(edge_index , x , pos)
        x = self.mlp2(x)
        x = self.lrelu(x + short_cut_x)

        f_p = x.view(B , N , -1)
        pos = pos.view(B,N,-1)
        rec = rec.view(B,N,-1)
        #pdb.set_trace()
        """
        f_p  for transformer  f_p concatenate f_v 
        pos  for points position 
        rec  for embeeding  query         
        """
        return f_p , pos , rec
    

    
class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, channels , return_mode:None):
        super().__init__(aggr="add")
        # spatial encoder 
        self.mlp_encoder = SharedMLP([10, channels // 2])  # 10 -> 32  
        
        # dgcnn encoder
        self.mlp_dgcnn = SharedMLP([3*(channels//2), channels]) # 32*3 
        # attention
        feature_dim = channels * 2 

        self.mlp_attention = SharedMLP([feature_dim, feature_dim], bias=False, act=None, norm=None)
        self.mlp_post_attention = SharedMLP([feature_dim, channels])


        self.return_mode = return_mode

        if self.return_mode == 'both':
            self.mlp_raw_features = SharedMLP([feature_dim , 4*channels])

    def forward(self, edge_index, x, pos):
       
        if self.return_mode == 'both':
            att_out = self.propagate(edge_index, x=x, pos=pos , mode = 'attention')  # N, d_out
            att_out = self.mlp_post_attention(att_out)  # N, d_out
            raw_out = self.propagate(edge_index , x , pos , mode = 'raw')
            raw_out = self.mlp_raw_features(raw_out)
            #pdb.set_trace()
            return att_out , raw_out
        else:
            att_out = self.propagate(edge_index, x=x, pos=pos , mode = 'attention')  # N, d_out
            att_out = self.mlp_post_attention(att_out)  # N, d_out
            return att_out

    def message(self, x_j: Tensor, x_i:Tensor,
                pos_i: Tensor, pos_j: Tensor, 
                index: Tensor , mode:str) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        #pdb.set_trace()
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance], dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N * K, 2d
        # Encode dgcnn

        edge_feature = torch.cat([x_i, x_j, x_j - x_i], dim=1)  # edge feature
        #pdb.set_trace()
        dgcnn_feature = self.mlp_dgcnn(edge_feature)

        #  fusion all features 

        local_features = torch.cat([dgcnn_feature , local_features ] ,dim=1)

        if mode == 'attention':
            # Attention will weight the different features of x
            # along the neighborhood dimension.
            att_features = self.mlp_attention(local_features)  # N * K, d_out
            att_scores = softmax(att_features, index=index)  # N * K, d_out
            return att_scores * local_features
        else: 
            return local_features  # N * K, d_out





# class AttentivePooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AttentivePooling, self).__init__()

#         self.score_fn = nn.Sequential(
#             nn.Linear(in_channels, in_channels, bias=False),
#             nn.Softmax(dim=-2)
#         )
#         self.mlp = MLP_LPFE(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

#     def forward(self, x):
#         r"""
#             Forward pass

#             Parameters
#             ----------
#             x: torch.Tensor, shape (B, d_in, N, K)

#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, 1)
#         """
#         # computing attention scores
#         #pdb.set_trace()
#         scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

#         # sum over the neighbors
#         features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

#         return self.mlp(features)




# class MLP_LPFE(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         stride=1,
#         transpose=False,
#         padding_mode='zeros',
#         bn=False,
#         activation_fn=None
#     ):
#         super(MLP_LPFE, self).__init__()

#         conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

#         self.conv = conv_fn(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding_mode=padding_mode
#         )
#         self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
#         self.activation_fn = activation_fn

#     def forward(self, input):
#         r"""
#             Forward pass of the network

#             Parameters
#             ----------
#             input: torch.Tensor, shape (B, d_in, N, K)

#             Returns
#             -------
#             torch.Tensor, shape (B, d_out, N, K)
#         """
#         x = self.conv(input)
#         if self.batch_norm:
#             x = self.batch_norm(x)
#         if self.activation_fn:
#             x = self.activation_fn(x)
#         return x
    

# class GSAModule(nn.Module):
#     def __init__(self, geometric_ind,in_channels_f_pv , hidden_dim , dropout = 0.1):
#         super(GSAModule, self).__init__()

#         self.hidden_dim = hidden_dim

#         # 特征变换层
#         self.to_qkv = nn.ModuleDict({
#             'q': nn.Linear(in_channels_f_pv, hidden_dim),
#             'k': nn.Linear(in_channels_f_pv, hidden_dim),
#             'v': nn.Linear(in_channels_f_pv, hidden_dim)
#         })
#         self.to_r = nn.Linear(geometric_ind, hidden_dim)

#         # 特征融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, in_channels_f_pv)
#         )

        
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(in_channels_f_pv)

#     def forward(self,point_voxel_features , geometric_encoding):
#         # 计算Q,K,V矩阵
#         #pdb.set_trace()
#         q = self.to_qkv['q'](point_voxel_features)
#         k = self.to_qkv['k'](point_voxel_features) 
#         v = self.to_qkv['v'](point_voxel_features)
#         r = self.to_r(geometric_encoding)

#           # 注意力计算  
#         attn = (torch.matmul(q, r.transpose(-2, -1)) + 
#                 torch.matmul(q, k.transpose(-2, -1))) / (self.hidden_dim ** 0.5)
        
#         # 归一化和dropout
#         attn = self.dropout(F.softmax(attn, dim=-1))
        
#         # 特征聚合和融合
#         out = torch.matmul(attn, v)
#         out = self.fusion(out)

#         out = self.norm(out + point_voxel_features) 

#         return out
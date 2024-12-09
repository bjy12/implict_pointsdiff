import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from copy import deepcopy
from einops import rearrange
#pdb.set_trace()
from model.modules.Encoder.image_wise.u_net import UNet
from model.modules.Encoder.fusion_wise.view_agg_points import ViewPoints_Aggregation
from model.modules.PV_module.multi_scale_volume_points import MultiScale_Points_Volume_Encoder
from model.modules.Decoder.triplane_points_implict import Multi_Scale_Triplane_Encoder , TriplaneImplictDecoder
#from model.models.diffusion_utilis.diffusion import IdensityDiffusion
from model.models.denoised.pointscloudtransformer import PointCloudTransformerModel
from model.models.denoised.pointnet2_with_pcld_condition import PointNet2CloudCondition
import pdb
def check_feature_range(x, name):
    print(f"{name} range: {x.min().item():.4f} to {x.max().item():.4f}")



def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]



class Points_View_Fusion_ImplictDiffuse(nn.Module):
    def __init__(self, image_encoder_config , aggregation_config , multi_scale_pv_encoder ,
                 triplane_encoder,triplane_decoder, denoised_config ,
                 ct_res ):
        super(Points_View_Fusion_ImplictDiffuse, self).__init__()
        # image_wise_porcess
        self.image_encoder = UNet(**image_encoder_config)
        self.fusion_view_aggregator = ViewPoints_Aggregation(**aggregation_config)
        #pdb.set_trace()
        self.ct_res = ct_res
        # points_wise_process
        self.multi_scale_pv_encoder = MultiScale_Points_Volume_Encoder(**multi_scale_pv_encoder)
        self.multi_scale_trip_encoder = Multi_Scale_Triplane_Encoder(**triplane_encoder)
        self.trip_imp_decoder = TriplaneImplictDecoder(**triplane_decoder)
        #pdb.set_trace()
        self.points_denoised_model = PointNet2CloudCondition(denoised_config)
  
        #diffusion_process 
        # denoise model
        #pdb.set_trace()
        # self.denoise_point_cloud_model = PointCloudTransformerModel(
        #                 num_layers=denoised_config['num_layers'],
        #                 in_channels=denoised_config['in_channels'],
        #                 out_channels=denoised_config['out_channels'],
        #                 embed_dim=denoised_config['embed_dim'],
        #                 **{k: v for k, v in denoised_config.items() 
        #                 if k not in ['num_layers', 'in_channels', 'out_channels', 'embed_dim']}
        #             )    
        #pdb.set_trace()
        # self.diffusion_model = IdensityDiffusion( diffusion_config=diffusion_config ,
        #                                           denoise_net=self.denoise_point_cloud_model, 
        #                                           denoise_type=denoise_type)
        #pdb.set_trace()
    def set_input(self , batch):
        #pdb.set_trace()
        projs = batch['projs']

        points = batch['points']

        points_gt = batch['points_gt']

        points_proj = batch['points_proj']

        view_feature = batch['view_feature']

        local_coords = batch['local_coord']

        return projs , points , points_gt , points_proj , view_feature , local_coords
    def image_wise_encoder(self, projs):
        b , m , c , w , h  = projs.shape
        projs = projs.reshape(b*m, c , w , h)
        proj_feats = self.image_encoder(projs)
        #pdb.set_trace()
        proj_feats = list(proj_feats) if type(proj_feats) is tuple else [proj_feats]
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            proj_feats[i] = proj_feats[i].reshape(b, m, c_, w_, h_) # B, M, C, W, H
            #check_feature_range(proj_feats[i] , f"proj_feats_scale_{i}")
        #pdb.set_trace()

        return proj_feats
    
    def project_points_(self, proj_feats ,points_proj):
        n_view = proj_feats[0].shape[1]
        # query view_specific features 
        p_list = []
        p_list = []
        for i in range(n_view):
            f_list = []
            for proj_f in proj_feats:
                feat = proj_f[:, i, ...] # B, C, W, H
                p = points_proj[:, i, ...] # B, N, 2
                p_feats = index_2d(feat, p) # B, C, N
                f_list.append(p_feats)
            p_feats = torch.cat(f_list, dim=1)
            p_list.append(p_feats)
        p_feats = torch.stack(p_list, dim=-1) # B, C, N, M
        p_feats = rearrange(p_feats , " b c n m -> b m n c")
        return p_feats
    

    def aggregation_points_f(self, view_feature , points_feats):

        aggregation_points = self.fusion_view_aggregator(view_feature , points_feats)

        return aggregation_points


    
    def points_wise_forward(self, aggreatation_points , global_coord , local_coord):
        #pdb.set_trace()
        """
        aggreatation_points: aggreatation points feature with view conditions  B N C
        global_coord: position of all ct grid  B N 3  for points branch get points features
        local_coord: position of crop ct grid  B N 3  for voxel branch  get voxel features 

        return : points_trip_f_implict  B N C  
        """
        multi_volume_points_f_stack , local_coord_stack , global_coord_stack = self.multi_scale_pv_encoder(aggreatation_points , global_coord, local_coord)
        #pdb.set_trace()
        #!TODO 
        #* multi_scale_triplane_f :a list include  multi level triplane feature     
        multi_scale_triplane_f = self.multi_scale_trip_encoder(multi_volume_points_f_stack , global_coord_stack) # 
        #pdb.set_trace()
        # keep order with random down sample 
        points_implict_condition = self.trip_imp_decoder(global_coord_stack  , multi_scale_triplane_f )

        return points_implict_condition, global_coord_stack

    def get_points_implict_wise_condition(self , projs , points_proj , view_feature , points , local_points):
        proj_feats = self.image_wise_encoder(projs) # B M C W H 
        #check_feature_range(proj_feats[0], "proj_feats")

        #pdb.set_trace()
        points_feats = self.project_points_(proj_feats , points_proj) # B M  N C
        
        #check_feature_range(points_feats, "points_feats")

        aggregation_points = self.aggregation_points_f(view_feature , points_feats)
        #pdb.set_trace()
        #check_feature_range(aggregation_points, "aggregation_points")

        points_implict , _ =  self.points_wise_forward(aggregation_points ,points , local_points)
        #check_feature_range(points_implict, "points_implict")

        return points_implict 
    
    def forward(self, projs , global_coords , points_proj , view_feature , local_coords ,noised_x ,ts):
        """
        input : type is torch tensor float32 
        projs: [b ,numbers_of_view , c , h , w ]  c(1) xray image  
        points: [b , numbers_of_points , 3]  3(x y z)  global coord 
        points_proj: [b , numbers_of_view , number of projected_coord , 2 ] 2(u v)
        view_feature: [b, numbers_of_view , number of points , c_v] c_v view feature (10 include postion, distance , ....)
        local_points: [b, numbers_of_points, 3 ] 3 (x y z ) local coords  for voxeliaztion 
        noised_x: [b , number_of_poitns , 3+1 ]  noised points-wise idensity  
        time_step: [b , time_steps]   
 
        """
        #pdb.set_trace()
        #projs, points , points_gt , points_proj , view_feature , local_coords = self.set_input(batch)
        #check_feature_range(projs, "projs_input")
        #check_feature_range(global_coords, "points_input")
        #check_feature_range(points_gt, "points_gt")
        #check_feature_range(noised_x, "noised_x")
        #check_feature_range(points_proj, "points_proj_input")
        #check_feature_range(view_feature, "view_feature")
        #check_feature_range(local_coords, "local_points")
        
        #pdb.set_trace()
        #step 1  make points_wise condition 
        condition_coord = global_coords.clone()
 
        points_implict_conditons = self.get_points_implict_wise_condition(projs , points_proj , 
                                                                                           view_feature , global_coords ,
                                                                                            local_coords)   
        #step 2 use condition guided points_wise denoised 
        #pdb.set_trace()
        points_implict_conditons = torch.cat([condition_coord , points_implict_conditons] , dim=2)
        loss = self.points_denoised_model(pointcloud = noised_x ,condition=points_implict_conditons, ts=ts)

        return loss









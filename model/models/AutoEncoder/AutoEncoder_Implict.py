import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.modules.Encoder.points_wise.RandLA import RandLAEncoder
from model.modules.Encoder.points_wise.dgcnn_cls import DGCNN_cls
from model.modules.Encoder.volume_wise.encoder import SRResnet
from model.modules.Decoder import coord_mlp 
from model.modules.Decoder.local_decoder import LocalDecoder
import pdb


class AutoEncoder_Implict_Points_CT(nn.Module):
    def __init__(self, encoder_name , decoder_name ,encoder_config , decoder_config ):
        super(AutoEncoder_Implict_Points_CT, self).__init__()
        #* encoder 

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name



        if encoder_name == "RandLA":
            self.encoder = RandLAEncoder(**encoder_config)
        if encoder_name == "DGCNN":
            self.encoder = DGCNN_cls(**encoder_config)
        if encoder_name == 'SRResnet':
            self.encoder = SRResnet(**encoder_config)    


       
        if decoder_name == "coord_mlp":
            self.decoder = coord_mlp.MLP(**decoder_config)
        if decoder_name == "local_decoder":
            self.decoder = LocalDecoder(**decoder_config)

    def get_input(self ,points , points_gt):

        coord = points
        coord[..., :3] -= 0.5 # [-0.5, 0.5]
        coord = rearrange(coord , ' b n c -> b c n')
        points = rearrange(points , 'b n c -> b c n')
        x = torch.cat([points, points_gt], dim=1)

        x = rearrange(x , ' b c n -> b n c')

        return x , coord    
    
    def encode(self , x ):
        latent_code = self.encoder(x)
        return latent_code
    

    def decode(self, x , coord):

        if self.decoder_name == 'local_decoder':
            #pdb.set_trace()
            coord = rearrange(coord , "b c n -> b n c")
            pred_intensity = self.decoder(coord , x )
            pred_intensity = pred_intensity.unsqueeze(1)
        if self.decoder_name == 'coord_mlp':
            #pdb.set_trace()
            feature_vector = F.grid_sample(x, coord.flip(-1).unsqueeze(1).unsqueeze(1),
                                           mode='bilinear',
                                           align_corners=False)
            feature_vector = feature_vector[:,:,0,0,:].permute(0,2,1)
            #pdb.set_trace()
            feature_vector_and_xyz_hr = torch.cat([feature_vector, coord], dim=-1)  # N×K×(3+feature_dim)
            #pdb.set_trace()
            N, K = feature_vector_and_xyz_hr.shape[:2]
            pred_intensity = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1) 


        
        return pred_intensity
    
    def set_input(self , coord , gt , ae_mode ):
        if ae_mode == "pc":
            #pdb.set_trace()
            gt = rearrange(gt , ' b c n -> b n c')
            x = torch.cat([coord , gt] , dim=-1)
            return x 
        if ae_mode == "volume":
            #! to do when volume type data 
            return 


    def forward(self, coord , gt , ae_mode ):
        # pdb.set_trace()
        # x , coord =  self.get_input(points , points_gt)
        #pdb.set_trace()
        x = self.set_input(coord , gt , ae_mode )
        #pdb.set_trace()
        latent_code = self.encode(x)
        #pdb.set_trace()
        #pdb.set_trace()
        #feature_map = feature_map.squeeze(-1)
        #coord = rearrange(coord , ' b c n -> b n c')
        #feature_map = rearrange(feature_map , ' b c n -> b n c')
        #feature_vector_coord = torch.cat([feature_map , coord], dim = -1) 
        #pdb.set_trace()
        pred_intensity = self.decode(latent_code,coord)
        #pdb.set_trace()

        #pdb.set_trace()

    
        return pred_intensity











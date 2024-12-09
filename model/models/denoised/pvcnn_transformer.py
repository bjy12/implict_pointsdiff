import torch 
import torch.nn as nn
import pdb
import numpy as np 
from model.modules.PV_module.pvcnn_utils import create_mlp_components , create_pointnet2_fp_modules , create_pointnet2_sa_components
from model.modules.PV_module.pvcnn import Attention


class PVCNN_Attention_Base(nn.Module):
    def __init__(self,
                num_classes: int, 
                embed_dim: int, 
                use_att: bool = True, 
                dropout: float = 0.1,
                extra_feature_channels: int = 3, 
                width_multiplier: int = 1, 
                voxel_resolution_multiplier: int = 1,
                sa_blocks: list = None,
                fp_blocks: list = None,
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.drop_out = dropout
        
        self.sa_blocks = sa_blocks
        self.fp_blocks = fp_blocks

        self.width_multiplier = width_multiplier  # default 1 
        #pdb.set_trace()

        self.in_channels = extra_feature_channels + 3   # extra_feature_channesl  1 + condtion_dim 
        # sa_layers   

        sa_layers , sa_in_channels ,channels_sa_features , _  = create_pointnet2_sa_components(
                sa_blocks=self.sa_blocks , #  ()
                extra_feature_channels=extra_feature_channels, # condition channels  dim 
                with_se = True ,
                embed_dim=embed_dim,
                use_att=use_att,
                dropout=dropout,
                width_multiplier= width_multiplier,
                voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        # for dowan sample feature 
        self.sa_layers = nn.ModuleList(sa_layers)


        # Additional global attention module
        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)
        
        # Only use extra features in the last fp module
        #pdb.set_trace()
        sa_in_channels[0] = extra_feature_channels + 3 
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, 
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim, 
            use_att=use_att, 
            dropout=dropout, 
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features, 
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True, 
            dim=2, 
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)


        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, inputs , t , local_coord , global_coord ):
        """
        input: [noised_idensity , condition_features ]  b , n , c1    c1 = 1 + condition_dim 
        local_coord: for pvcnn voxelization   [b n 3] x y z 
        global_coord: for get position information  [b n 3] x y z  

        return :
        output : [b , n , out_channels] 
        """
        #pdb.set_trace()
        t_emb = get_timestep_embedding(self.embed_dim, t, inputs.device).float()
        #pdb.set_trace()
        t_emb = self.embedf(t_emb)[:, :, None].expand(-1, -1, inputs.shape[1])

        #features = torch.concatenate([global_coord , inputs] , dim=2)
        #pdb.set_trace()
        inputs = inputs.permute(0,2,1)
        features = inputs.clone()
        #pdb.set_trace()
        features_0 = features.clone()
        #global_coord = global_coord[:,:,:3].contiguous()
        global_coord = global_coord.permute(0,2,1)
        local_coord = local_coord.permute(0,2,1)
        #pdb.set_trace()
        
        global_coord_list = []
        in_feature_list = []
        local_coord_list = []
        #pdb.set_trace()
        for i , sa_blocks in enumerate(self.sa_layers):
            #! 需要保证每次local coord 的顺序 和  global coord的顺序保持一致
            #! local_coord 是为了体素化特征的提取准备的坐标 voxelization过程需要注意 
            in_feature_list.append(features)
            global_coord_list.append(global_coord)
            local_coord_list.append(local_coord)
            if i == 0:
                features , global_coord , t_emb , local_coord = sa_blocks((features , global_coord, t_emb , local_coord ))
                #pdb.set_trace()
            else:
                #pdb.set_trace()
                features , global_coord , t_emb , local_coord = sa_blocks((torch.cat([features,t_emb],dim= 1), global_coord ,t_emb , local_coord))
        #pdb.set_trace()
        #Replace the input features 
        #in_feature_list[0] = features_0[:,3:,:].contiguous()
        in_feature_list[0] = features_0.contiguous()

        #Apply global attention layer 
        if self.global_att is not None:
            features = self.global_att(features)
        #pdb.set_trace()
        #Upscaling layer
        for fp_idx , fp_blocks in enumerate(self.fp_layers):
            #pdb.set_trace()
            print("fp_idx : " , fp_idx)
            features , global_coord, t_emb = fp_blocks(
                (
                    global_coord_list[-1-fp_idx], # revers coords list from above 
                    global_coord, # global point coordinates
                    torch.cat([features, t_emb] , dim=1), # keep concatenating upsampled features with timesteps
                    in_feature_list[-1 - fp_idx], # reverse features list of from above
                    t_emb ,# original timestep embedding
                    local_coord_list[-1 - fp_idx] 
                )
            )

        output = self.classifier(features)
        #pdb.set_trace()
        output = output.permute(0,2,1)
        return output

        

        







def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for 
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb
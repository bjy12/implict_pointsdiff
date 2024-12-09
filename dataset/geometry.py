import numpy as np
from copy import deepcopy
import pdb



class Geometry(object):
    def __init__(self, config):
        self.v_res = config['nVoxel'][0]    # ct scan
        self.p_res = config['nDetector'][0] # projections
        self.v_spacing = np.array(config['dVoxel'])[0]    # mm
        self.p_spacing = np.array(config['dDetector'])[0] # mm
        # NOTE: only (res * spacing) is used

        self.DSO = config['DSO'] # mm, source to origin
        self.DSD = config['DSD'] # mm, source to detector

    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        d1 = self.DSO
        d2 = self.DSD

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        return d_points  
      
    def get_dis_plane_points_and_o(self, points, angle):

        d1 = self.DSO
        d2 = self.DSD
        #pdb.set_trace()
        points = deepcopy(points).astype(float)
        N , _  = points.shape
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T

        rot_1_p_0 = points
   
        coeff_1 = (d2) / (d1 - rot_1_p_0[:,0])
        d_p_1 = rot_1_p_0[:,[2,1]] * coeff_1[:,None]

        dis_plane_from_o = np.array([-(d2-d1)])
        dis_plane_from_o = dis_plane_from_o[None,:]
        dis_plane_from_o = dis_plane_from_o.repeat(N,0)
        #pdb.set_trace()
        plane_points  = np.concatenate([dis_plane_from_o , d_p_1],axis=1)
        source = np.array([d1,0,0])

        #pdb.set_trace()
        dis_plane_points_from_source = np.linalg.norm(plane_points - source, axis=1)

        dis_rotated_points_from_source = np.linalg.norm(rot_1_p_0 - source, axis=1)
   
        distance_ratio = dis_rotated_points_from_source / dis_plane_points_from_source 

        direction_source2planepoints = plane_points - source

        vector_norms = np.linalg.norm(direction_source2planepoints , axis=1 , keepdims=True)

        norm_vectors = direction_source2planepoints / ( vector_norms + 1e-6 )




        return distance_ratio  , norm_vectors       

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points


import numpy as np
import trimesh

from tqdm import tqdm

from probreg import cpd

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

import warnings
warnings.filterwarnings("ignore")


from GHD import GHD_config, GHDmesh, Normal_iterative_GHDmesh
from einops import rearrange, einsum, repeat

from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss




from losses import *
from ops.mesh_geometry import *

class GHD_Cardiac(GHDmesh):
    def __init__(self, cfg):
        super(GHD_Cardiac, self).__init__(cfg)
        self.tri_mesh = trimesh.Trimesh(self.base_shape.verts_packed().detach().cpu().numpy(), 
                                            self.base_shape.faces_packed().detach().cpu().numpy())
        self.device = cfg.device
        
    
    def dice_evaluation(self, target_positives, target_negatives):
        """
        evaluate the dice loss of the current mesh and the target point cloud
        target_positives: (N, 3)
        target_negatives: (N, 3)
        return: the dice loss
        """
        DiceLoss = BinaryDiceLoss()
        with torch.no_grad():
            current_mesh = self.rendering()
            occupancy_pos = Winding_Occupancy(current_mesh, target_positives, max_v_per_call=100)
            occupancy_pos = torch.sigmoid((occupancy_pos-0.5)*100).view(1, -1) 
            occupancy_neg = Winding_Occupancy(current_mesh, target_negatives,max_v_per_call=100)
            occupancy_neg = torch.sigmoid((occupancy_neg-0.5)*100).view(1, -1)
            occupancy_sample = torch.cat([occupancy_pos, occupancy_neg], dim=-1)
            occupancy_gt = torch.cat([torch.ones_like(occupancy_pos), torch.zeros_like(occupancy_neg)], dim=-1)

        return DiceLoss(occupancy_sample, occupancy_gt).item()


    def fitting2target(self, target_positives, 
                    target_negatives, 
                    target_mesh=None,
                    loss_dict = {'Loss_occupancy':1, 'Loss_Laplacian':0.01, 'Loss_rigid':0.01},
                    lr_start=0.01, 
                    num_iter=2000, 
                    num_sample=2000,
                    NP_ratio=3,
                    if_reset=True, 
                    if_fit_R=False, 
                    if_fit_s=True, 
                    if_fit_T=True,
                    Distance_weighted=1.,
                    record_convergence=False):
        """
        the main function of the GHD_Cardiac, which is used to morph the bi-ventricle mesh to the target bi-ventricle point cloud
        target_positives: (N, 3) the target point cloud of the positive samples contained in the target mesh
        target_negatives: (N, 3) the target point cloud of the negative samples excluded by the target mesh
        target_mesh: the target mesh (default: None, if None, the chamfer distance will not be calculated). Ignored if 'Loss_Chamfer_P0' is not in loss_dict.keys()
        num_iter: the number of iterations
        num_sample: the number of  positive samples
        NP_ratio: the ratio of the number of  samples to the negative number of  positive samples
        if_reset: if reset the GHD parameters
        if_fit_R: if fit the rotation matrix
        if_fit_s: if fit the scale
        if_fit_T: if fit the translation
        Distance_weighted: if consider the distance-weighted DVS loss, which is used to avoid the sensitivity on the boundary points
        record_convergence: if record the convergence process
        return: the morphed mesh
        """
        if if_reset:
            self.reset_GHD_param()
        
        Rigid_Losser = Rigid_Loss(self.base_shape)
        # Rigid_Losser = Rigid_Loss(ico_sphere(4, device=paraheart.device))
        DiceLoss = BinaryDiceLoss()

        optim_params = [self.GHD_param]
        if if_fit_R:
            optim_params.append(self.R)
        if if_fit_s:
            optim_params.append(self.s)
        if if_fit_T:
            optim_params.append(self.T)

        
        if Distance_weighted > 0:
            # if Distance_weighted is True, we will use the knn_points to get the distances

            dist_p2n = knn_points(target_positives.view(1, -1, 3),
                                     target_negatives.view(1, -1, 3), 
                                     K=1)[0].view(-1)
            dist_n2p = knn_points(target_negatives.view(1, -1, 3),
                                        target_positives.view(1, -1, 3), 
                                        K=1)[0].view(-1)
            dist_mean = torch.cat([dist_p2n, dist_n2p], dim=0).mean()
            dist_weights_p2n = 1 - torch.exp(-dist_p2n**2 / (Distance_weighted*dist_mean.mean()**2 + 1e-6))
            dist_weights_p2n = dist_weights_p2n/dist_weights_p2n.mean()
            
            dist_weights_n2p = 1 - torch.exp(-dist_n2p**2 / (Distance_weighted*dist_mean.mean()**2 + 1e-6))
            dist_weights_n2p = dist_weights_n2p/dist_weights_n2p.mean()

        else:
            dist_weights_p2n = torch.ones_like(target_positives[:, 0])
            dist_weights_n2p = torch.ones_like(target_negatives[:, 0])


        optmizer = torch.optim.AdamW(optim_params, lr=lr_start)
        scheduler = torch.optim.lr_scheduler.StepLR(optmizer, step_size=100, gamma=0.8)

        thinknesser = MeshThickness(r=0.2, num_bundle_filtered=50, innerp_threshold=0.6, num_sel=10)

        Loss_GHD = 0.
        loop = tqdm(range(num_iter))

        if record_convergence:
            convergence = []

            Loss_dict_list = []

        

        for j in loop:
            optmizer.zero_grad()
            current_mesh = self.rendering()

            # differientiable occupancy + dice loss
            Loss_values = {}
            if 'Loss_occupancy' in loss_dict.keys():
                pos_idx = torch.randperm(target_positives.shape[0])[:num_sample]
                neg_idx = torch.randperm(target_negatives.shape[0])[:int(num_sample*NP_ratio)]

                samples_pos = target_positives[pos_idx]
                pos_weights = dist_weights_p2n[pos_idx]
                pos_weights = pos_weights/ pos_weights.mean()


                samples_neg = target_negatives[neg_idx]
                neg_weights = dist_weights_n2p[neg_idx]
                neg_weights = neg_weights/ neg_weights.mean()

                weights = torch.cat([pos_weights, neg_weights], dim=0).view(1, -1)

                occupancy_pos = Winding_Occupancy(current_mesh, samples_pos)
                # occupancy_pos = Winding_Occupancy(current_mesh, target_positives)
                occupancy_pos = torch.sigmoid((occupancy_pos-0.5)*10).view(1, -1) 
                occupancy_neg = Winding_Occupancy(current_mesh, samples_neg)
                # occupancy_neg = Winding_Occupancy(current_mesh, target_negatives)
                occupancy_neg = torch.sigmoid((occupancy_neg-0.5)*10).view(1, -1)
                occupancy_sample = torch.cat([occupancy_pos, occupancy_neg], dim=-1)
                occupancy_gt = torch.cat([torch.ones_like(occupancy_pos), torch.zeros_like(occupancy_neg)], dim=-1)

                Loss_values['Loss_occupancy'] = DiceLoss(occupancy_sample, occupancy_gt, weight=weights)

            # chamfer distance P0, N1
            if 'Loss_Chamfer_P0' in loss_dict.keys():
                assert (target_mesh is not None), "target_mesh is required for chamfer distance"
                sample_num = 2000
                current_lv_sample, current_lv_normals_sample = sample_points_from_meshes(current_mesh, sample_num, return_normals=True)
                mesh_gt_lv_sample, mesh_gt_lv_normals_sample = sample_points_from_meshes(target_mesh, sample_num, return_normals=True)
                Loss_Chamfer_P0, Loss_Chamfer_N1 = chamfer_distance(current_lv_sample, mesh_gt_lv_sample, x_normals=current_lv_normals_sample, y_normals=mesh_gt_lv_normals_sample)
                Loss_values['Loss_Chamfer_P0'] = Loss_Chamfer_P0
                if 'Loss_Chamfer_N1' in loss_dict.keys():
                    Loss_values['Loss_Chamfer_N1'] = Loss_Chamfer_N1

            ## regularizations
            # laplacian smoothing
            if 'Loss_Laplacian' in loss_dict.keys():
                Loss_values['Loss_Laplacian'] = mesh_laplacian_smoothing(current_mesh, method='uniform')

            # equal edge length
            if 'Loss_equaledge' in loss_dict.keys():
                Loss_values['Loss_equaledge'] = uniform_length(current_mesh)

            # local similarity
            if 'Loss_rigid' in loss_dict.keys():
                Loss_values['Loss_rigid'] = Rigid_Losser.forward(current_mesh.verts_packed(), if_elastic=True)

            # total loss
            if 'Loss_normal_consistency' in loss_dict.keys():
                Loss_values['Loss_normal_consistency'] = mesh_normal_consistency(current_mesh)

            if 'Loss_thickness' in loss_dict.keys():
                thinkness_dict,thinkness, _, sign = thinknesser.forward(current_mesh)
                # softmin_thinkness = F.softmin(thinkness*sign, dim=-1)

                # thinkness = thinkness*sign

                # softmin_thinkness = (softmin_thinkness*thinkness*sign).sum(dim=-1)
                mask = torch.where(thinkness.abs()>0.4, torch.zeros_like(thinkness), torch.ones_like(thinkness))
                signed = torch.sign(sign)

                Loss_values['Loss_thickness'] = (F.relu(0.04 - thinkness*signed) + F.relu(0.01 - thinkness_dict*signed))*mask

                Loss_values['Loss_thickness'] = Loss_values['Loss_thickness'].mean()

                Loss_values['Loss_thickness'] += (1e-4/(sign**2+1e-6)*mask).mean()
                 # 4mm is the minimal thickness of the heart

                # Loss_values['Loss_thickness'] += F.exp(-thinkness*sign).mean()

            Loss_GHD = 0.
            for key in loss_dict.keys():
                Loss_GHD += loss_dict[key]*Loss_values[key]


            Loss_GHD.backward()
            optmizer.step()
            scheduler.step()

            if record_convergence:
                Loss_dict = {key:Loss_values[key].item() for key in Loss_values.keys()}
                Loss_dict_list.append(Loss_dict)
                if j<1000 and j%10==0:
                    convergence.append(current_mesh.verts_packed().detach())

            loop.set_description('Total Loss %.4f'%(Loss_GHD.item()))
                    
        print('fittings done, the final loss is %.6f'%(Loss_GHD.item()))
        Loss_dict = {key:Loss_values[key].item() for key in Loss_values.keys()}
        if record_convergence:
            return convergence, Loss_dict_list 
        
        return current_mesh, Loss_dict

if __name__ == "__main__":
    cfg = GHD_config(base_shape_path="canonical/Standard_LV_800.obj",
                 num_basis=6**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
                device='cuda:0',
                if_nomalize=True, if_return_scipy=True)

    paraheart = GHD_Cardiac(cfg)
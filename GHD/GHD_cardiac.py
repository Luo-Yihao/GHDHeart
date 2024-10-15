
import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes 


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
        self.bi_ventricle = trimesh.load_mesh(cfg.bi_ventricle_path)
        self.lv_ventricle = trimesh.Trimesh(self.base_shape.verts_packed().detach().cpu().numpy(), self.base_shape.faces_packed().detach().cpu().numpy())
        self.device = cfg.device
        
    def global_registration_biv(self, target_pcl):
        """ 
        the global registration of the bi-ventricle mesh and the target bi-ventricle point cloud,
        which gurantee the initial affine parameters is basically correct and avoid the ambiguity caused by the symmetry of the left ventricle
        target_pcl: (N, 3)
        return: the affine parameters (update the self.R, self.s, self.T automatically)
        """

        num_points = target_pcl.shape[0]
        point_cloud_canonical_bi = trimesh.sample.volume_mesh(self.bi_ventricle, num_points)

        R_matrix = axis_angle_to_matrix(self.R)[0].detach().cpu().numpy()
        s = self.s[0].detach().cpu().numpy()
        T = self.T[0].detach().cpu().numpy()
        param_dict = {'rot':R_matrix, 'scale':s, 't':T}
        rgd_cpd = cpd.RigidCPD(point_cloud_canonical_bi, tf_init_params=param_dict, update_scale=True)
        tf_param, _, _ = rgd_cpd.registration(target_pcl)
        R, s, T =tf_param.rot, tf_param.scale, tf_param.t
        param_dict = {'rot':R, 'scale':s, 't':T}
        

        self.R = nn.Parameter(matrix_to_axis_angle(torch.from_numpy(R).float().to(self.device)).unsqueeze(0))
        self.s = nn.Parameter(torch.tensor([s], device=self.device).float().unsqueeze(0))
        self.T = nn.Parameter(torch.from_numpy(T).float().to(self.device).unsqueeze(0))

        return param_dict
    
    def global_registration_lv(self, target_pcl):
        """ 
        the global registration of the left ventricle mesh and the target left ventricle point cloud,
        which gurantee the initial affine parameters is basically correct and avoid the ambiguity caused by the symmetry of the left ventricle
        target_pcl: (N, 3)
        return: the affine parameters (update the self.R, self.s, self.T automatically)
        """

        num_points = target_pcl.shape[0]
        point_cloud_canonical_lv = trimesh.sample.volume_mesh(self.lv_ventricle, num_points)

        R_matrix = axis_angle_to_matrix(self.R)[0].detach().cpu().numpy()
        s = self.s[0].detach().cpu().numpy()
        T = self.T[0].detach().cpu().numpy()
        param_dict = {'rot':R_matrix, 'scale':s, 't':T}
        rgd_cpd = cpd.RigidCPD(point_cloud_canonical_lv, tf_init_params=param_dict, update_scale=True)
        tf_param, _, _ = rgd_cpd.registration(target_pcl)
        R, s, T =tf_param.rot, tf_param.scale, tf_param.t
        param_dict = {'rot':R, 'scale':s, 't':T}
        

        self.R = nn.Parameter(matrix_to_axis_angle(torch.from_numpy(R).float().to(self.device)).unsqueeze(0))
        self.s = nn.Parameter(torch.tensor([s], device=self.device).float().unsqueeze(0))
        self.T = nn.Parameter(torch.from_numpy(T).float().to(self.device).unsqueeze(0))

        return param_dict

    def rendering_bi_ventricle(self):
        """
        render the bi-ventricle mesh by the current Affine parameters
        """
        output_trimesh = self.bi_ventricle.copy()
        R_matrix = axis_angle_to_matrix(self.R)[0].detach().cpu().numpy()
        s = self.s[0].detach().cpu().numpy()
        T = self.T[0].detach().cpu().numpy()
        new_verts = (self.bi_ventricle.vertices @ R_matrix.transpose(-1,-2))*s + T

        
        output_trimesh.vertices = new_verts
        return output_trimesh
    
    def dice_evaluation(self, target_positives, target_negatives):
        """
        evaluate the dice loss of the current mesh and the target point cloud
        target_positives: (N, 3)
        target_negatives: (N, 3)
        return: the dice loss
        """
        DiceLoss = BinaryDiceLoss()
        with torch.no_grad():
            current_lv_mesh = self.rendering()
            occupancy_lv = Winding_Occupancy(current_lv_mesh, target_positives, max_v_per_call=100)
            occupancy_lv = torch.sigmoid((occupancy_lv-0.5)*100).view(1, -1) 
            occupancy_outoflv = Winding_Occupancy(current_lv_mesh, target_negatives,max_v_per_call=100)
            occupancy_outoflv = torch.sigmoid((occupancy_outoflv-0.5)*100).view(1, -1)
            occupancy_sample = torch.cat([occupancy_lv, occupancy_outoflv], dim=-1)
            occupancy_gt = torch.cat([torch.ones_like(occupancy_lv), torch.zeros_like(occupancy_outoflv)], dim=-1)
        
        return DiceLoss(occupancy_sample, occupancy_gt).item()


    def morphing2lvtarget(self, target_positives, 
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
                          record_convergence=False):
        """
        the main function of the GHD_Cardiac, which is used to morph the bi-ventricle mesh to the target bi-ventricle point cloud
        target_pcl: (N, 3)
        num_iter: the number of iterations
        num_sample: the number of  positive samples
        NP_ratio: the ratio of the number of  samples to the negative number of  positive samples
        if_reset: if reset the GHD parameters
        if_fit_R: if fit the rotation matrix
        if_fit_s: if fit the scale
        if_fit_T: if fit the translation
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
            current_lv_mesh = self.rendering()


            # differientiable occupancy + dice loss
            Loss_values = {}
            if 'Loss_occupancy' in loss_dict.keys():
                occupancy_lv = Winding_Occupancy(current_lv_mesh, target_positives[torch.randperm(target_positives.shape[0])[:num_sample]])
                # occupancy_lv = Winding_Occupancy(current_lv_mesh, target_positives)
                occupancy_lv = torch.sigmoid((occupancy_lv-0.5)*100).view(1, -1) 
                occupancy_outoflv = Winding_Occupancy(current_lv_mesh, target_negatives[torch.randperm(target_negatives.shape[0])[:num_sample*NP_ratio]])
                # occupancy_outoflv = Winding_Occupancy(current_lv_mesh, target_negatives)
                occupancy_outoflv = torch.sigmoid((occupancy_outoflv-0.5)*100).view(1, -1)
                occupancy_sample = torch.cat([occupancy_lv, occupancy_outoflv], dim=-1)
                occupancy_gt = torch.cat([torch.ones_like(occupancy_lv), torch.zeros_like(occupancy_outoflv)], dim=-1)
                Loss_values['Loss_occupancy'] = DiceLoss(occupancy_sample, occupancy_gt)

            # chamfer distance P0, N1
            if 'Loss_Chamfer_P0' in loss_dict.keys():
                assert (target_mesh is not None), "target_mesh is required for chamfer distance"
                sample_num = 2000
                current_lv_sample, current_lv_normals_sample = sample_points_from_meshes(current_lv_mesh, sample_num, return_normals=True)
                mesh_gt_lv_sample, mesh_gt_lv_normals_sample = sample_points_from_meshes(target_mesh, sample_num, return_normals=True)
                Loss_Chamfer_P0, Loss_Chamfer_N1 = chamfer_distance(current_lv_sample, mesh_gt_lv_sample, x_normals=current_lv_normals_sample, y_normals=mesh_gt_lv_normals_sample)
                Loss_values['Loss_Chamfer_P0'] = Loss_Chamfer_P0
                if 'Loss_Chamfer_N1' in loss_dict.keys():
                    Loss_values['Loss_Chamfer_N1'] = Loss_Chamfer_N1

            ## regularizations
            # laplacian smoothing
            if 'Loss_Laplacian' in loss_dict.keys():
                Loss_values['Loss_Laplacian'] = mesh_laplacian_smoothing(current_lv_mesh, method='uniform')

            # equal edge length
            if 'Loss_equaledge' in loss_dict.keys():
                Loss_values['Loss_equaledge'] = uniform_length(current_lv_mesh)

            # local similarity
            if 'Loss_rigid' in loss_dict.keys():
                Loss_values['Loss_rigid'] = Rigid_Losser.forward(current_lv_mesh.verts_packed(), if_elastic=True)

            # total loss
            if 'Loss_normal_consistency' in loss_dict.keys():
                Loss_values['Loss_normal_consistency'] = mesh_normal_consistency(current_lv_mesh)

            if 'Loss_thickness' in loss_dict.keys():
                thinkness_dict,thinkness, _, sign = thinknesser.forward(current_lv_mesh)
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
                    convergence.append(current_lv_mesh.verts_packed().detach())

            loop.set_description('Total Loss %.4f'%(Loss_GHD.item()))
                    
        print('fittings done, the final loss is %.6f'%(Loss_GHD.item()))
        Loss_dict = {key:Loss_values[key].item() for key in Loss_values.keys()}
        if record_convergence:
            return convergence, Loss_dict_list 
        return current_lv_mesh, Loss_dict

if __name__ == "__main__":
    cfg = GHD_config(base_shape_path="metadata/canonical_worldcoord_mmwhs.obj",
                 num_basis=6**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
                device='cuda:0',
                if_nomalize=True, if_return_scipy=True, bi_ventricle_path='/home/yihao/data/ParaHearts/data/biventricle_mmwhs.obj')

    paraheart = GHD_Cardiac(cfg)
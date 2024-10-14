import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))

from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from .rigid_deform import RigidLoss
from ops.graph_operators import Laplacain, Normal_consistence

from ops.torch_warping import feature_align_with_turbulence
from torch_geometric.utils import to_undirected


from typing import Union

class Mesh_loss(nn.Module):
    def __init__(self, mesh_std: Meshes, sample_num=5000, resolution=128, length=2, ):
        super(Mesh_loss, self).__init__()
        self.sample_num = sample_num
        self.mesh_std = mesh_std

        cotweight_std, _ = cot_laplacian(mesh_std.verts_packed(), mesh_std.faces_packed())
        
        self.connection_std = cotweight_std.coalesce().indices()
        self.cotweight_std = cotweight_std.coalesce().values()


        self.connection_std, self.cotweight_std = to_undirected(self.connection_std, edge_attr=self.cotweight_std)
        
        self.rigidloss = RigidLoss()
        self.lap = Laplacain()
        self.normal_consistence = Normal_consistence()

        self.edge_lenth = torch.norm(self.mesh_std.verts_packed().index_select(0,self.connection_std[0]) - self.mesh_std.verts_packed().index_select(0,self.connection_std[1]),dim=-1).mean()

        self.resolution = resolution
        self.length = length


    
    
    def forward(self, meshes_scr: Meshes, trg: Union[Meshes, torch.Tensor], loss_list:dict, B=1):



        
        loss_dict = {}


        sample_scr, normals_scr = sample_points_from_meshes(meshes_scr,self.sample_num, return_normals=True)
        
        if isinstance(trg, Meshes):
            sample_trg, normals_trg = sample_points_from_meshes(trg,self.sample_num, return_normals=True)
            loss_p0, loss_n1 = chamfer_distance(sample_scr, sample_trg, x_normals=normals_scr, y_normals=normals_trg)
        else:
            sample_trg = trg
            loss_p0, loss_n1 = chamfer_distance(sample_scr, sample_trg, x_normals=None, y_normals=None)
            loss_n1 = 1e-5


        if 'loss_p0' in loss_list:
            loss_dict['loss_p0'] = loss_p0

        if 'loss_n1' in loss_list:
            loss_dict['loss_n1'] = loss_n1


        if 'loss_laplacian' in loss_list:
        
            laplacain_vect = mesh_laplacian_smoothing(meshes_scr, method="cot")
    
            loss_dict['loss_laplacian'] = laplacain_vect


        if 'loss_edge' in loss_list:
            loss_dict['loss_edge'] = mesh_edge_loss(meshes_scr,self.edge_lenth.to(meshes_scr.device))

        if 'loss_consistency' in loss_list:
            loss_dict['loss_consistency'] = mesh_normal_consistency(meshes_scr)
            
        if 'loss_rigid' in loss_list:
            

            verts_scr = meshes_scr.verts_padded()
            verts_std = self.mesh_std.verts_packed()

            loss_dict['loss_rigid'] = torch.zeros_like(verts_std[:,:1])

            for i in range(B):
                rigid_i = self.rigidloss.forward(verts_scr[i],verts_std, self.connection_std, self.cotweight_std)
                loss_dict['loss_rigid'] += rigid_i
            loss_dict['loss_rigid'] = loss_dict['loss_rigid'].mean()/B

        return loss_dict

            

if __name__ == "__main__":
    from pytorch3d.io import load_objs_as_meshes
    B = 2
    device = torch.device("cpu")
    mesh_std = load_objs_as_meshes(['/home/yihao/data/PINN_GCN/data/canonical.obj'],device=device)[0]
    N = mesh_std.verts_packed().shape[0]
    meshes_scr = load_objs_as_meshes(['/home/yihao/data/PINN_GCN/data/canonical.obj']*B,device=device)
    meshes_scr = meshes_scr.offset_verts(torch.randn(B*N,3).to(device))
    meshes_trg = meshes_scr.offset_verts(torch.randn(B*N,3).to(device))

    gt3d = torch.randint(0,2,(B,2,128,128,128)).float().to(device)
    #ico_sphere(3, device)
    mesh_loss = Mesh_loss(mesh_std,50)
    loss_dict = mesh_loss(meshes_scr, meshes_trg, ['loss_p0','loss_n1','loss_consistency','loss_laplacian','loss_edge','loss_seg_align','loss_rigid'],gt3d)
    print(loss_dict)
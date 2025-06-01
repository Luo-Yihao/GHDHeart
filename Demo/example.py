# %%
# import pyvista as pv
# pv.start_xvfb(wait=0)
# pv.set_jupyter_backend('html')

import os
import sys
sys.path.append(os.path.join('..', '.'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import cubify, cot_laplacian, sample_points_from_meshes, knn_points, knn_gather, norm_laplacian, taubin_smoothing
from pytorch3d.loss import chamfer_distance
from pytorch3d.utils import ico_sphere

from torch_geometric.utils import degree, to_undirected, to_dense_adj, get_laplacian, add_self_loops
from torch_geometric.data import Data
# from torch_geometric.transforms import gdc
from torch_scatter import scatter

import numpy as np

import trimesh

import warnings

from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix


from data_process.dataset_real_scaling import Example_Simple_dataset, point_cloud_extractor
from ops.graph_operators import NativeFeaturePropagation, LaplacianSmoothing


from tqdm import tqdm

from probreg import cpd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


import warnings
warnings.filterwarnings("ignore")

from GHD import GHD_config, GHDmesh, Normal_iterative_GHDmesh
from GHD.GHD_cardiac import GHD_Cardiac


from einops import rearrange, einsum, repeat

from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss


from losses import *
from ops.mesh_geometry import *
import data.data_utils as dut
import ops.medical_related as med

from data.dataset import Example_Simple_dataset

def main():
    
    GHD_basis = None
    GHD_eigval = None
    if canonical_size == 1705:
        base_shape_path = '../canonical_shapes/wholeheart_canonical/LV.obj'
        if num_basis == 6**2:
            GHD_basis = torch.load('../canonical_shapes/wholeheart_canonical/GHD_basis_4CH_B36.pt')['lv']
            GHD_eigval = torch.load('../canonical_shapes/wholeheart_canonical/GHD_eigval_4CH_B36.pt')['lv']
            print("loading GHD basis and eigenvalues from ../canonical_shapes/wholeheart_canonical/GHD_basis_4CH_B36.pt")

    elif canonical_size == 2000:
        base_shape_path = '../canonical_shapes/Standard_LV_2000.obj'
    elif canonical_size == 800:
        base_shape_path = '../canonical_shapes/Standard_LV_800.obj'
    elif canonical_size == 4055:
        base_shape_path = '../canonical_shapes/Standard_LV_4055.obj'
    else:
        raise ValueError('Please specify canonical size as 2000, 800 or 4055')
    
    print("Number of Faces: ", canonical_size)



    # base_shape_path = 'metadata/Standard_LV.obj'
    # bi_ventricle_path = 'metadata/Standard_BiV.obj'
    

    cfg = GHD_config(base_shape_path=base_shape_path,
                num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
                device=device,
                GH_eigval= GHD_eigval,
                GH_eigvec= GHD_basis,
                if_nomalize=True, 
                if_return_scipy=True)

    paraheart = GHD_Cardiac(cfg) 
    root_path = os.path.dirname(os.path.realpath('.'))

    root_path = os.path.join(root_path,'data_example')

    example_data = Example_Simple_dataset(dataset_path=root_path)

    if target == 'ED':
        data = example_data[0]
    elif target == 'ES':
        data = example_data[1]
    else:
        raise ValueError('Please specify target as ED or ES')
  

    affine_tem = data['affine'].to(device).unsqueeze(0).to(device)
    label_tem = data['seg_gt'].to(device).unsqueeze(0).to(device)

    ## Mesh Gt from marchingcubes
    mesh_gt_lv = cubify((label_tem==example_data.label_value[2]).squeeze(1).float(), 0.48)
    mesh_gt_lv = mesh_gt_lv.update_padded(mesh_gt_lv.verts_padded().bmm(affine_tem[:,:3,:3].transpose(1,2)) + affine_tem[:,:3,3].unsqueeze(1).expand(-1, mesh_gt_lv.verts_padded().shape[1], -1))


    # mesh_gt_lv = cubify((label_tem==example_data.label_value[2]).squeeze(1).float(), 0.48)
    # mesh_gt_lv = taubin_smoothing(mesh_gt_lv, 0.1, 0.5, num_iter=20)
    
    B = label_tem.shape[0]
    ## -------------- Data augmentation -----------------##
    coordinate_map_tem = dut.get_coord_map_3d_normalized(label_tem.shape[-3:], affine_tem)


    Z_rv, Y_rv, X_rv = torch.where(label_tem[0, 0]==example_data.label_value[1])
    Z_lv, Y_lv, X_lv = torch.where(label_tem[0, 0]==example_data.label_value[2])
    Z_cav, Y_cav, X_cav = torch.where(label_tem[0, 0]==example_data.label_value[3])
    Z_bg, Y_bg, X_bg = torch.where(label_tem[0, 0]==example_data.label_value[0])
    
    Pt_rv = coordinate_map_tem[0, Z_rv, Y_rv, X_rv]
    Pt_lv = coordinate_map_tem[0, Z_lv, Y_lv, X_lv]
    Pt_cav = coordinate_map_tem[0, Z_cav, Y_cav, X_cav]
    Pt_bg = coordinate_map_tem[0, Z_bg, Y_bg, X_bg]



    points_bi = torch.cat([Pt_rv, Pt_lv], dim=0)
    points_lv = Pt_lv
    points_outoflv = torch.cat([Pt_rv, Pt_cav, Pt_bg], dim=0)

    geom_dict = med.get_4chamberview_frame(Pt_cav, Pt_lv, Pt_rv)
    inital_affine = geom_dict['target_affine']

    bbox_lv = torch.stack([Pt_lv.min(dim=0)[0]-0.05, Pt_lv.max(dim=0)[0]+0.05], dim=-1)

    points_outoflv_in_bbox = points_outoflv[(points_outoflv[:,0]>bbox_lv[0,0]) & (points_outoflv[:,0]<bbox_lv[0,1]) & (points_outoflv[:,1]>bbox_lv[1,0]) & (points_outoflv[:,1]<bbox_lv[1,1]) & (points_outoflv[:,2]>bbox_lv[2,0]) & (points_outoflv[:,2]<bbox_lv[2,1])]


    paraheart.R = matrix_to_axis_angle(inital_affine[...,:3,:3].to(paraheart.device)).view(paraheart.R.shape)
    paraheart.T = inital_affine[...,:3,3].to(paraheart.device).view(paraheart.T.shape)


    ## ------------------ Global registration -----------------##
    sample_num = 2000

    sample_lv = points_lv[np.random.choice(points_lv.shape[0], sample_num, replace=False)]
    paraheart.global_registration(sample_lv.detach().cpu().numpy())

 


    # ----- GHD fitting ------
    convergence, Loss_dict_list  = paraheart.fitting2target(points_lv, points_outoflv_in_bbox, loss_dict 
                                = {'Loss_occupancy':1,  'Loss_Laplacian':0.001, 'Loss_thickness':0.001},
                                lr_start=lr, num_iter=iter, if_reset=True, if_fit_R=False, if_fit_s=True, 
                                if_fit_T=True, record_convergence=True, Distance_weighted= 1.)


    ## --------Visualization------

    coordinate_map_np = coordinate_map_tem[0].detach().cpu().numpy()

    import pyvista as pv
    pv.start_xvfb(wait=0)

    coordinate_map_np = coordinate_map_tem[0].detach().cpu().numpy()

    pl = pv.Plotter(notebook=False, shape=(1,3), off_screen=True)
    pl.subplot(0,0)
    interval = 1
    if label_tem.shape[-3] > 20:
        interval = 5
    if label_tem.shape[-3] > 256:
        interval = 10
    for i in range(0, label_tem.shape[-3], interval):

        x, y, z = coordinate_map_np[i,...,0], coordinate_map_np[i,...,1], coordinate_map_np[i,...,2]

        grid = pv.StructuredGrid(x, y, z)

        color_gt = (label_tem[0,0,i].cpu().numpy().T.flatten()).astype(np.float32)
        
        color_opacity = np.ones_like(color_gt)*0.2

        color_opacity[color_gt == 0] = 0.1

        pl.add_mesh(grid, scalars = color_gt, cmap = 'Accent_r',
                    show_scalar_bar = False, opacity = color_opacity,
                    lighting=False)
        
    out_ghd_mesh = paraheart.rendering()

    # trimesh_current_bi = paraheart.rendering_bi_ventricle()
    # trimesh_current_bi = pv.wrap(trimesh_current_bi)
    # pl.add_mesh(trimesh_current_bi, color='blue', opacity=0.2)

    # trimesh_gt_lv = trimesh.Trimesh(mesh_gt_lv.verts_packed().detach().cpu().numpy(), mesh_gt_lv.faces_packed().detach().cpu().numpy())
    # pl.add_mesh(trimesh_gt_lv, color='lightgreen', opacity=0.1)

    trimesh_current_lv = trimesh.Trimesh(out_ghd_mesh.verts_packed().detach().cpu().numpy(), 
                                         out_ghd_mesh.faces_packed().detach().cpu().numpy())
    pl.add_mesh(trimesh_current_lv, color='lightblue', opacity=0.8, 
                show_edges=True, show_vertices=False, lighting=False)

    pl.add_mesh(pv.Box(bounds=[-1, 1, -1, 1, -1, 1]).outline(), color='black')
    pl.add_text(target, font_size=10, position='upper_edge')

    # pl.add_points(lv_cavity_center.cpu().numpy(), color='red', point_size=10)

    pl.subplot(0,1)
    # light = pv.Light(position=(0.4, -0.2, 1), color='white', intensity=0.8)

    # pl.add_light(light)

    pl.add_mesh(trimesh_current_lv, color='lightblue', 
                opacity=1, show_edges=True, show_vertices=1, lighting=True,
                point_size=3, line_width=0.6)
    

    pl.add_text('GHD (Ours)', font_size=10, position='upper_edge')
    pl.add_text('Verts: %d \nFaces: %d' % (trimesh_current_lv.vertices.shape[0], trimesh_current_lv.faces.shape[0]), 
                font_size=10, position='lower_right')

    pl.subplot(0,2)
    trimesh_gt_lv = trimesh.Trimesh(mesh_gt_lv.verts_packed().detach().cpu().numpy(), mesh_gt_lv.faces_packed().detach().cpu().numpy())
    pl.add_mesh(trimesh_gt_lv, color='lightcoral', opacity=1, show_edges=False, show_vertices=0)
    pl.add_text('MarchingCubes', font_size=10, position='upper_edge')
    pl.add_text('Verts: %d \nFaces: %d' % (trimesh_gt_lv.vertices.shape[0], trimesh_gt_lv.faces.shape[0]), 
                font_size=10, position='lower_right')

    ## make dir if not exists
    if not os.path.exists('output'):
        os.makedirs('output', exist_ok=True)

    visual = pl.screenshot('output/visualization_'+target+'.png', 
                           window_size=[int(1024*2.5), 1024])
    

        
    Dice = 1- paraheart.dice_evaluation(points_lv, points_outoflv_in_bbox)

    print("The final 3D dice %.4f"%(Dice*100)+"%")

    Loss_occupancy = [loss['Loss_occupancy'] for loss in Loss_dict_list]
    Loss_occupancy = np.array(Loss_occupancy)
    plt.plot(Loss_occupancy)
    plt.title('DVS convergening')
    plt.savefig('output/convergence_'+target+'.png')
    print("The final convergence figure is saved as Demo/output/convergence_"+target+".png")
    
    out_ghd_mesh = paraheart.rendering()
    trimesh_current_lv = trimesh.Trimesh(out_ghd_mesh.verts_packed().detach().cpu().numpy(), out_ghd_mesh.faces_packed().detach().cpu().numpy())


    trimesh_current_lv.export('output/Lv_result_'+target+'.obj')
    print("The final mesh is saved as Demo/output/Lv_result_"+target+".obj")
    print("The final visualization about comparison between GHD and MarchingCubes is saved as Demo/output/visualization_"+ target+".png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--csize', type=int, default=1705, help='alternative canonical meshes with number of faces 1705, 2000, 800, 4055')
    parser.add_argument('--num_basis', type=int, default=6**2, help='number of basis GHD, default is 9*9')
    parser.add_argument('--target', type=str, default='ED')
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3*5)
    args = parser.parse_args()
    device = args.device
    num_basis = args.num_basis
    target = args.target
    iter = args.iter
    lr = args.lr
    canonical_size = args.csize

    main()



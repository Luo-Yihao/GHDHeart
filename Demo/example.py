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


from data_process.dataset_real_scaling import UKBB_dataset, MMWHS_dataset, ACDC_dataset, CCT48_dataset

from einops import rearrange, einsum, repeat

from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss


from losses import *
from ops.mesh_geometry import *

import pickle


def main():

# %%
    base_shape_path = '../canonical_shapes/Standard_LV_2000.obj'
    bi_ventricle_path = '../canonical_shapes/Standard_BiV.obj'

    # base_shape_path = 'metadata/Standard_LV.obj'
    # bi_ventricle_path = 'metadata/Standard_BiV.obj'

    cfg = GHD_config(base_shape_path=base_shape_path,
                num_basis=num_basis, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
                device='cuda:3',
                if_nomalize=True, if_return_scipy=True, 
                bi_ventricle_path=bi_ventricle_path)

    paraheart = GHD_Cardiac(cfg) # 


    # load initial orientation according to dataset

    with open('../canonical_shapes/ukbb_init_affine.pkl', 'rb') as f:
        initial_orientation = pickle.load(f)

    R = initial_orientation[:3,:3].astype(np.float32)
    T = initial_orientation[:3,3].astype(np.float32)

    paraheart.R = matrix_to_axis_angle(torch.from_numpy(R).to(paraheart.device)).view(paraheart.R.shape)
    paraheart.T = torch.from_numpy(T).to(paraheart.device).view(paraheart.T.shape)

    # %%
    root_path = os.path.dirname(os.path.realpath('.'))

    root_path = os.path.join(root_path,'data_example')

    example_data = Example_Simple_dataset(dataset_path=root_path)

    

    # %%
    dataloader_mmwhs = DataLoader(example_data, batch_size=1, shuffle=False)

    LaplacianSmoother = LaplacianSmoothing()

    if target == 'ED':
        data = example_data[0]
    elif target == 'ES':
        data = example_data[1]
    else:
        raise ValueError('Please specify target as ED or ES')
  
    seg_gt = data['seg_gt'].unsqueeze(0).to(device)


    point_list = point_cloud_extractor(seg_gt,  [ 0., 1., 2., 4.], data['window'], spacing=200, coordinate_order = 'zyx')

    points_lv = point_list[2]
    points_outoflv = torch.cat([point_list[0], point_list[1], point_list[3]], dim=0)
    points_bi = torch.cat([point_list[-2], point_list[-1]], dim=0)

    bbox_lv = torch.stack([points_lv.min(dim=0)[0], points_lv.max(dim=0)[0]], dim=0).T
    rescale = 1.1
    bbox_lv_center =  bbox_lv.mean(-1)
    bbox_lv = torch.stack([bbox_lv_center-rescale*(bbox_lv_center-bbox_lv[:,0]), bbox_lv_center+rescale*(bbox_lv[:,1]-bbox_lv_center)], dim=-1)
    points_outoflv_in_bbox = points_outoflv[(points_outoflv[:,0]>bbox_lv[0,0]) & (points_outoflv[:,0]<bbox_lv[0,1]) & (points_outoflv[:,1]>bbox_lv[1,0]) & (points_outoflv[:,1]<bbox_lv[1,1]) & (points_outoflv[:,2]>bbox_lv[2,0]) & (points_outoflv[:,2]<bbox_lv[2,1])]


    # %%
    sample_num = 2000

    mesh_gt_bi_sample = points_bi.detach().cpu().numpy()[np.random.choice(points_bi.shape[0], sample_num, replace=False)]
    paraheart.global_registration_biv(mesh_gt_bi_sample)


    sample_lv = points_lv[np.random.choice(points_lv.shape[0], sample_num, replace=False)]
    paraheart.global_registration_lv(sample_lv.detach().cpu().numpy())

 


    # %%

    # sample_outoflv = points_outoflv_in_bbox[np.random.choice(points_outoflv_in_bbox.shape[0], sample_num*5, replace=False)]

    convergence, Loss_dict_list  = paraheart.morphing2lvtarget(points_lv, points_outoflv_in_bbox, loss_dict 
                                = {'Loss_occupancy':1,  'Loss_Laplacian':0.001, 'Loss_thickness':0.001},
                                lr_start=1e-3, num_iter=iter, if_reset=True, if_fit_R=False, if_fit_s=True, if_fit_T=True, record_convergence=True)

    Dice = 1- paraheart.dice_evaluation(points_lv, points_outoflv_in_bbox)
    out_ghd_mesh = paraheart.rendering()
    trimesh_current_lv = trimesh.Trimesh(out_ghd_mesh.verts_packed().detach().cpu().numpy(), out_ghd_mesh.faces_packed().detach().cpu().numpy())

    ## make dir if not exists
    if not os.path.exists('output'):
        os.makedirs('output', exist_ok=True)
    trimesh_current_lv.export('output/Lv_result_'+target+'.obj')
    print("The final 3D dice %.4f"%(Dice*100)+"%")
    print("The final mesh is saved as Demo/output/Lv_result_"+target+".obj")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_basis', type=int, default=7**2)
    parser.add_argument('--target', type=str, default='ED')
    parser.add_argument('--iter', type=int, default=500)
    args = parser.parse_args()
    device = args.device
    num_basis = args.num_basis
    target = args.target
    iter = args.iter

    main()



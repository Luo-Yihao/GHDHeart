import torch
import torch.nn.functional as F

import numpy as np


def random_affine_matrix(rot_factor = 0.2, trans_factor = 0, scale_factor = 0.0, B=1):

    scale_rand = torch.randn(B,1)*scale_factor+1
    scale_rand = torch.clip(scale_rand, 1-scale_factor, 1+scale_factor)

    scale_rand = scale_rand.repeat(1,3)
    scale_rand = torch.diag_embed(scale_rand)

    rot_factor = np.clip(rot_factor, 0, 1)

    # random rotation by expm of skew-symmetric matrix
    augmented_affine = torch.randn((B,3,3))
    augmented_affine = augmented_affine - augmented_affine.transpose(-1,-2)
    augmented_affine = torch.linalg.matrix_exp(rot_factor*augmented_affine)

    augmented_affine = scale_rand.matmul(augmented_affine)

    # random translation
    augmented_affine = torch.cat([augmented_affine, trans_factor*(torch.rand((B,3,1))*2-1.0)], dim = -1)
    augmented_affine = torch.cat([augmented_affine, torch.tensor([0,0,0,1]).view(1,1,4).repeat(B,1,1)], dim = -2)

    return augmented_affine

def one_hot_sparse(y_ij, edge_index_i):
    y_ij_onehot_0 = torch.sparse_coo_tensor(torch.stack([edge_index_i,torch.arange(0,edge_index_i.shape[0]).to(edge_index_i.device)]),y_ij[:,0],size=(edge_index_i.max().item()+1,edge_index_i.shape[0]),requires_grad=True)

    y_ij_onehot_1 = torch.sparse_coo_tensor(torch.stack([edge_index_i,torch.arange(0,edge_index_i.shape[0]).to(edge_index_i.device)]),y_ij[:,1],size=(edge_index_i.max().item()+1,edge_index_i.shape[0]),requires_grad=True)

    y_ij_onehot_2 = torch.sparse_coo_tensor(torch.stack([edge_index_i,torch.arange(0,edge_index_i.shape[0]).to(edge_index_i.device)]),y_ij[:,2],size=(edge_index_i.max().item()+1,edge_index_i.shape[0]),requires_grad=True)

    y_ij_onehot = torch.stack([y_ij_onehot_0,y_ij_onehot_1,y_ij_onehot_2],dim=-1)

    return y_ij_onehot
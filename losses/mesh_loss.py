import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import  cot_laplacian, sample_points_from_meshes, knn_points
from pytorch3d.loss import chamfer_distance

from einops import rearrange, einsum, repeat

from torch_geometric.utils import degree, to_undirected, to_dense_adj, get_laplacian, add_self_loops
# from torch_geometric.transforms import gdc
from torch_scatter import scatter



def compute_thickness(Mesh_scr:Meshes, num_sample=10000):
    TB_sample_1 = torch.cat(sample_points_from_meshes(Mesh_scr, num_sample, return_normals=True), dim=-1)
    TB_vertices = torch.cat([Mesh_scr.verts_padded(), Mesh_scr.verts_normals_padded()], dim=-1)
    TB_sample_1 = torch.cat([TB_sample_1, TB_vertices], dim=-2)
    TB_sample_2 = TB_sample_1.clone()

    TB_sample_2[:,:,3:] = -TB_sample_2[:,:,3:]

    knn = knn_points(TB_sample_1, TB_sample_2, K=2, return_sorted=True,return_nn=True)
    knn_nn = knn[-1][:,:,1:,:]

    thickness_vect = knn_nn[:,:,0,:3] - TB_sample_1[...,0:3]
    thickness = torch.norm(thickness_vect, dim=-1)

    signed = ((thickness_vect*TB_sample_1[...,3:]).sum(-1)/(thickness+1e-8))
    
    return thickness, signed

def uniform_length(Mesh_scr):
    """
    Args:
        Mesh: Meshes
    """
    edge_indx = Mesh_scr.edges_packed()
    edge_length = torch.norm(Mesh_scr.verts_packed()[edge_indx[:,0]] - Mesh_scr.verts_packed()[edge_indx[:,1]], dim=1)
    loss_edge = torch.square(edge_length.mean() - edge_length)
    edge_weight = F.softmax(loss_edge, dim=0)
    loss_out = torch.sum(edge_weight*loss_edge)/(torch.sum(edge_weight)+1e-6)
    loss_out += edge_length.std()
    return loss_out/(edge_length.mean()+1e-6)


def Soft_Hausdorff(Mesh_scr, Mesh_trg, sample_num=None):
    """"
    Hausdorff distance between two meshes with softmin and softmax to be differentiable
    Args:
        Mesh_scr: Meshes
        Mesh_trg: Meshes
        sample_num: int
    """ 

    if sample_num is None:
        num_points = Mesh_scr.verts_packed().shape[0] + Mesh_trg.verts_packed().shape[0]
    else:
        num_points = sample_num

    sample_scr = sample_points_from_meshes(Mesh_scr, num_points, return_normals=False)
    sample_trg = sample_points_from_meshes(Mesh_trg, num_points, return_normals=False)
                                                               
    sample_scr = sample_scr.view(1, -1, 3)
    sample_trg = sample_trg.view(1, -1, 3)


    P0_distance, _ = chamfer_distance(sample_scr, sample_trg,
                    point_reduction=None, batch_reduction=None)
    P0_src2trg, P0_trg2src = P0_distance

    # softmin 
    Softmin_P0_src2trg = F.softmin(P0_src2trg, dim=-1)
    Softmin_P0_trg2src = Softmin_P0_src2trg/Softmin_P0_src2trg.sum(dim=-1, keepdim=True)
    Softmin_P0_trg2src = F.softmin(P0_trg2src, dim=-1)
    Softmin_P0_src2trg = Softmin_P0_trg2src/Softmin_P0_trg2src.sum(dim=-1, keepdim=True)
    P0_min_src2trg = torch.sum(Softmin_P0_src2trg * P0_src2trg, dim=-1, keepdim=True)
    P0_min_trg2src = torch.sum(Softmin_P0_trg2src * P0_trg2src, dim=-1, keepdim=True)

    # Hausdorff
    Softmax_Hausdorff = F.softmax(torch.cat([P0_min_src2trg, P0_min_trg2src], dim=-1), dim=-1)
    Softmax_Hausdorff = Softmax_Hausdorff/Softmax_Hausdorff.sum(dim=-1, keepdim=True)
    Hausdorff_dis = torch.sum(Softmax_Hausdorff * torch.cat([P0_min_src2trg, P0_min_trg2src], dim=-1), dim=-1)

    return Hausdorff_dis

def closest_verts_dist(Mesh_scr):
    closest_verts_dist = (knn_points(Mesh_scr.verts_padded(), Mesh_scr.verts_padded(), K=2, return_sorted=True)[0][...,-1:]).view(-1)
    closest_verts_dist_min = torch.min(closest_verts_dist)
    closest_verts_dist_inv = closest_verts_dist_min / (closest_verts_dist+1e-6)
    closest_verts_softmax = F.softmax(closest_verts_dist_inv, dim=-1)
    closest_verts_softmax = closest_verts_softmax/(closest_verts_softmax.sum()+1e-6)
    closest_verts_min_inv = torch.sum(closest_verts_softmax * closest_verts_dist, dim=-1, keepdim=True)
    return closest_verts_min_inv

class Rigid_Loss(nn.Module):
    """
    As rigid as possible loss for the mesh, refer to the paper "As rigid as possible surface modeling"
    """
    def __init__(self, meshes_src):
        super().__init__()
        self.meshes_src = meshes_src
        self.cot_laplacian_weights = cot_laplacian(meshes_src.verts_packed(), meshes_src.faces_packed().long())[0]
        self.edges_src = self.cot_laplacian_weights._indices()
        self.cot_laplacian_weights = self.cot_laplacian_weights._values()
        self.undirected_edges_src, self.cot_laplacian_weights = to_undirected(self.edges_src, self.cot_laplacian_weights, meshes_src._V)
        self.degrees_src = degree(self.undirected_edges_src[0], meshes_src._V)
        self.max_degree_src = int(self.degrees_src.max().item())
        # complement the degree to the max degree by adding self loops
        add_self_loops = self.max_degree_src - self.degrees_src
        add_edges_list = []
        for i in range(len(add_self_loops)):
            num_self_loops = int(add_self_loops[i].item())
            if num_self_loops > 0:
                add_edges_list.append(torch.tensor([[i, i]] * num_self_loops, device=meshes_src.device))
        add_edges = torch.cat(add_edges_list, dim=0)
        self.undirected_edges_src_eq = torch.cat([self.undirected_edges_src, add_edges.T], dim=-1)
        added_weights = torch.zeros((add_edges.shape[0],), device=meshes_src.device)
        self.cot_laplacian_weights = torch.cat([self.cot_laplacian_weights, added_weights], dim=0)
        # sort undirected_edges_src_eq
        self.undirected_edges_src_eq = self.undirected_edges_src_eq[:,torch.sort(self.undirected_edges_src_eq[0])[1]]
        self.neighborhood_indices = self.undirected_edges_src_eq[1].view(meshes_src._V, self.max_degree_src)
        # gather the neighborhood coordinates with the scatter indices
        self.neighborhood_coords_src = meshes_src.verts_packed()[self.undirected_edges_src_eq[1], :]
        self.neighborhood_coords_src = rearrange(self.neighborhood_coords_src, '(v d) c -> v d c', v=meshes_src._V, d=self.max_degree_src, c=3)
        # gather the weights
        self.neighborhood_weights = self.cot_laplacian_weights[self.undirected_edges_src_eq[1]].view(meshes_src._V, self.max_degree_src)
    
    def forward(self, new_verts_coords, if_elastic=False):
        """
        new_verts_coords: (V, 3), V == meshes_src._V
        """
        neighborhood_coords_trg = new_verts_coords[self.undirected_edges_src_eq[1], :]
        neighborhood_coords_trg = rearrange(neighborhood_coords_trg, '(v d) c -> v d c', v=self.meshes_src._V, d=self.max_degree_src, c=3)

        # normalize the neighborhood coordinates
        neighborhood_coords_src = self.neighborhood_coords_src - self.meshes_src.verts_packed().unsqueeze(1)
        neighborhood_coords_trg = neighborhood_coords_trg - new_verts_coords.unsqueeze(1)

        
        if if_elastic:
            # compute the cross-covariance matrix
            XtX = torch.bmm(neighborhood_coords_src.transpose(-2, -1), neighborhood_coords_src)
            YtY = torch.bmm(neighborhood_coords_trg.transpose(-2, -1), neighborhood_coords_trg)
            # dim of tangent space = 2 (trace of the cross-covariance matrix)
            trace_XtX = torch.diagonal(XtX, dim1=-2, dim2=-1).sum(-1, keepdim=True)/2
            trace_XtX = trace_XtX.unsqueeze(-1)
            trace_YtY = torch.diagonal(YtY, dim1=-2, dim2=-1).sum(-1, keepdim=True)/2
    
            trace_YtY = trace_YtY.unsqueeze(-1)
            neighborhood_coords_src = neighborhood_coords_src / (trace_XtX+1e-6).sqrt()
            neighborhood_coords_trg = neighborhood_coords_trg / (trace_YtY+1e-6).sqrt()

        # least squares solution of the orthogonal Procrustes problem
        XtY = torch.bmm(neighborhood_coords_src.transpose(-2, -1), neighborhood_coords_trg)
        U, S, V = torch.svd(XtY+1e-6*torch.eye(3, device=new_verts_coords.device).unsqueeze(0))

        R_iT = V.matmul(U.transpose(-2, -1))
        Y_hat = neighborhood_coords_src.matmul(R_iT)

        # compute the loss
        Diff_Y = (Y_hat - neighborhood_coords_trg).norm(dim=-1)

        # sum the loss with the cotangent weights
        loss_rigid = (Diff_Y * self.neighborhood_weights).sum()/ (self.neighborhood_weights.sum()+1e-6)

        return loss_rigid



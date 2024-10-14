import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from ops.torch_algebra import one_hot_sparse


class RigidLoss(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow='target_to_source')

    def forward(self, x, y, edge_index, weight=None):
        edge_index, weight_undir = to_undirected(edge_index, edge_attr=weight)
        weight_undir_mean = scatter(weight_undir, edge_index[0], dim=0, reduce="add")
        weight_undir_mean = weight_undir_mean.index_select(0,edge_index[0])
        weight_undir = weight_undir/(weight_undir_mean+1e-6)

        
        # h = torch.cat([self.lin(x), self.lin(y)], dim=-1)

        h_prime = self.propagate(edge_index, x=x, y=y, weight_undir=weight_undir)
        
        return h_prime


    def message(self, x_i, x_j, y_i, y_j, edge_index_i, weight_undir):

        # x_Ni_mean = scatter(x_j, edge_index_i, dim=0, reduce="mean")
        # x_Ni_mean = x_Ni_mean.index_select(0,edge_index_i)

        # x_ii = (x_i - x_Ni_mean)
        # x_ij = (x_j - x_Ni_mean)
        x_ij = (x_j - x_i)

        # y_Ni_mean = scatter(y_j, edge_index_i, dim=0, reduce="mean")
        # y_Ni_mean = y_Ni_mean.index_select(0,edge_index_i)

        # y_ii = (y_i - y_Ni_mean)
        # y_ij = (y_j- y_Ni_mean)
        y_ij = (y_j- y_i)
    


        # x_ii_onehot = one_hot_sparse(x_ii, edge_index_i)
        # y_ii_onehot = one_hot_sparse(y_ii, edge_index_i)
        
        # XtX_onehot = torch.bmm(x_ii_onehot.transpose(-1,-2),x_ii_onehot.to_dense())
        # YtY_onehot = torch.bmm(y_ii_onehot.transpose(-1,-2),y_ii_onehot.to_dense())
        # traceX = torch.diagonal(XtX_onehot,dim1=-2,dim2=-1).sum(dim=-1)
        # traceY = torch.diagonal(YtY_onehot,dim1=-2,dim2=-1).sum(dim=-1)

        # traceX = torch.sqrt(traceX/3.0).index_select(0,edge_index_i)
        # traceY = torch.sqrt(traceY/3.0).index_select(0,edge_index_i)
        

        # y_ij_onehot = one_hot_sparse(y_ij/traceY.unsqueeze(-1), edge_index_i) # N*E*3
        # x_ij_onehot = one_hot_sparse(x_ij/traceX.unsqueeze(-1), edge_index_i)# N*E*3

        x_ij_onehot = one_hot_sparse(x_ij, edge_index_i)
        y_ij_onehot = one_hot_sparse(y_ij, edge_index_i)
        XtX_onehot = torch.bmm(x_ij_onehot.transpose(-1,-2),x_ij_onehot.to_dense())
        YtY_onehot = torch.bmm(y_ij_onehot.transpose(-1,-2),y_ij_onehot.to_dense())
        traceX = torch.diagonal(XtX_onehot,dim1=-2,dim2=-1).sum(dim=-1)
        traceY = torch.diagonal(YtY_onehot,dim1=-2,dim2=-1).sum(dim=-1)
        traceX = torch.sqrt(torch.abs(traceX)/2.0)
        traceY = torch.sqrt(torch.abs(traceY)/2.0)

        traceY = traceY.index_select(0,edge_index_i)
        traceX = traceX.index_select(0,edge_index_i)

        x_ij = x_ij/(traceX.unsqueeze(-1)+1e-6)
        y_ij = y_ij/(traceY.unsqueeze(-1)+1e-6)
        
        x_ij_onehot = one_hot_sparse(x_ij, edge_index_i)
        y_ij_onehot = one_hot_sparse(y_ij, edge_index_i)


        # XtY_onehot = torch.bmm(x_ij_onehot.transpose(-1,-2),y_ij_onehot.to_dense())
        XtY_onehot = torch.bmm(x_ij_onehot.transpose(-1,-2),y_ij_onehot.to_dense())


        U, S, VT = torch.linalg.svd(XtY_onehot+1e-6*torch.eye(3,device=XtY_onehot.device,dtype=torch.float32))


        R_iT = (U.matmul(VT))

        R_iT = R_iT.index_select(0,edge_index_i) # E*3*3

        rigid_error_ij = y_ij.unsqueeze(-2) - torch.bmm(x_ij.unsqueeze(-2),R_iT) #E*1*3

        # rigid_error_ij = rigid_error_ij.sum(dim=0).to_dense()
        rigid_error_ij = torch.norm(rigid_error_ij, dim=-1, keepdim=False)*weight_undir.unsqueeze(-1)


        # rigid_error_ij = rigid_error_ij.index_select(0,edge_index_i)

        return rigid_error_ij


if __name__ == "__main__":
    conv = RigidLoss()
    x = torch.rand(4, 3)
    y = torch.rand(4, 3)
    from ops.torch_algebra import random_affine_matrix
    R = random_affine_matrix()
    print('R:',R.shape)

    z = 10*x.matmul(R[0,:3,:3])+R[0,:3,3].view(1,3)

    print(z.shape)

    edge_index = torch.tensor(
        [[0,0, 1, 0, 1, 2], [1,3, 2, 2, 3,3]], dtype=torch.long)
    weight = torch.tensor([1,2,3,4,5,6], dtype=torch.float)
    x = conv(x,z+0*torch.randn_like(z), edge_index,weight)

    print(x)
 

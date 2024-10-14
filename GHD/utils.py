import torch
from scipy.sparse import coo_matrix
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from torch_geometric.utils import to_undirected, add_self_loops, get_laplacian

def laplician_symetric(weights_value, connection, N_verts, device, if_nomalize=True, if_return_scipy=True):
    """
    Construct the symmetric laplacian matrix from the weights and connection

    weights_value: the value of the weights (N_edges,)
    connection: the connection of the weights (2, N_edges)
    N_verts: the number of vertices

    """


    # symmetrize the adjacency matrix
    weights_adj = torch.sparse_coo_tensor(indices=connection , values=weights_value, size=(N_verts, N_verts), dtype=torch.float32, device=device)
    weights_adj = weights_adj + weights_adj.transpose(0, 1)

    # get the degree matrix
    weights_sum = torch.sparse.sum(weights_adj, dim=-1).to_dense().view(-1)
    weights_degree = torch.sparse_coo_tensor(indices=torch.stack([torch.arange(N_verts), torch.arange(N_verts)]), values=weights_sum, size=(N_verts, N_verts), dtype=torch.float32, device=device)

    if if_nomalize == True:
        weights_adj = weights_adj * ((torch.pow(weights_sum+1e-8, -0.5)).unsqueeze(-1))*((torch.pow(weights_sum+1e-8, -0.5)).unsqueeze(-2))
        # symmetrize the normalized adjacency matrix
        weights_adj = (weights_adj + weights_adj.transpose(0, 1))/2.0

        weights_degree = torch.sparse_coo_tensor(indices=torch.stack([torch.arange(N_verts), torch.arange(N_verts)]), values=torch.ones(N_verts), size=(N_verts, N_verts), dtype=torch.float32, device=device)


    # Laplacian = D - W
    weights_lap = weights_degree - weights_adj


    if if_return_scipy == True:
        weights_lap = weights_lap.coalesce()
        weights_lap = coo_matrix((weights_lap.values().cpu().numpy(), (weights_lap.indices()[0].cpu().numpy(), weights_lap.indices()[1].cpu().numpy())))

    return weights_lap




def get_mix_laplacian(mesh: Meshes, mix_laplacian_tradeoff: dict, if_return_scipy=True, if_nomalize=True):
    """
    get the mix laplacian matrix
    """

    device = mesh.device
    N_verts = mesh.verts_packed().shape[0]

    cot_weights, _ = cot_laplacian(mesh.verts_packed(), mesh.faces_packed())
    # be careful about the connection shuffling during the cot_laplacian, not equal to the edge_packed()
    connection = cot_weights.coalesce().indices().detach()

    # for the cot laplacian (mainly for the laplacian eigenmaps)
    cot_weights = cot_weights.coalesce().values().detach()
    cot_lap = laplician_symetric(cot_weights, connection, N_verts, device, if_nomalize=if_nomalize, if_return_scipy=if_return_scipy)

    edge_lenth = torch.norm(torch.index_select(mesh.verts_packed(), 0, connection[0]) 
                                - torch.index_select(mesh.verts_packed(), 0, connection[1]), dim=-1).detach()

    # for the distance laplacian
    distant_weights = torch.exp(-torch.pow(edge_lenth, 2)/2.0)
    distant_lap = laplician_symetric(distant_weights, connection, N_verts, device, if_nomalize=if_nomalize, if_return_scipy=if_return_scipy)

    # for the standard laplacian (the regularization term)
    stander_lap = laplician_symetric(torch.ones_like(cot_weights), connection, N_verts, device, if_nomalize=if_nomalize, if_return_scipy=if_return_scipy)

    
    mix_laplacian = cot_lap * mix_laplacian_tradeoff['cotlap'] + distant_lap * mix_laplacian_tradeoff['dislap'] + stander_lap * mix_laplacian_tradeoff['stdlap']

    return mix_laplacian, cot_lap, distant_lap, stander_lap
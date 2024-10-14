import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch3d.structures import Meshes


class EdgeDetection2D(nn.Module):
    def __init__(self):
        super(EdgeDetection2D, self).__init__()
        # Define the Sobel operator kernels
        self.horizontal_kernel = nn.Parameter(torch.tensor([[[[-1., 0., 1.],
                                                              [-2., 0., 2.],
                                                              [-1., 0., 1.]]]], dtype=torch.float32), requires_grad=False)
        
        self.vertical_kernel = nn.Parameter(torch.tensor([[[[-1., -2., -1.],
                                                            [ 0.,  0.,  0.],
                                                            [ 1.,  2.,  1.]]]], dtype=torch.float32), requires_grad=False)
        
        # Set the kernels in a convolution layer without bias
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=False)
        # Initialize the weights of the convolution layer with the Sobel kernels
        with torch.no_grad():
            self.conv.weight[0] = self.horizontal_kernel
            self.conv.weight[1] = self.vertical_kernel

    def forward(self, x):
        # Apply the edge detection convolution
        x = self.conv(x)
        # You may also consider using the magnitude of the gradients
        x = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x

def real_coord_3D(img, affine_matrix_torch, output_size= (256,256,256), abs_value=False):

    B, C, D, W, H = img.shape
    
    device = img.device

    affine_matrix = affine_matrix_torch.clone()
    det = torch.det(affine_matrix[:,0:3,0:3])
    scale = torch.pow(torch.abs(det), 1/3)
    # disregrad the translation for keeping the center
    affine_matrix[:,0:3,-1:] = 0

    grid_trans = affine_matrix.inverse()[:,0:3,:]*scale.view(B,1,1)

    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[-3], output_size[-2], output_size[-1])))
    
    grid = grid.to(device)

    warp_img = F.grid_sample(img, grid)

    if abs_value:
        warp_img = torch.where(warp_img>0.5, torch.ones_like(warp_img), torch.zeros_like(warp_img))
        
    return warp_img

def warp_img_torch_3D(img, transform_matrix, output_size, if_center_align=False, abs_value=False):
    # please pay exact attention on the order of H and W,
    # and the normalization of the grid in Torch, but not in OpenCV
    device = img.device
    B, C, D, H, W = img.shape
    T = torch.Tensor([[2 / (W-1), 0, 0, -1],
                      [0, 2 / (H-1), 0, -1],
                      [0, 0, 2 / (D-1), -1],
                      [0, 0, 0, 1]]).to(device).repeat(B,1,1)
    
    S = torch.Tensor([[2 / (output_size[2]-1), 0, 0, -1],
                    [0, 2 / (output_size[1]-1), 0, -1],
                    [0, 0, 2 / (output_size[0]-1), -1],
                    [0, 0, 0, 1]]).to(device).repeat(B,1,1)

    grid_trans = torch.matmul(T,torch.matmul(transform_matrix,S).inverse())[:,0:3,:]

    # M_torch = torch.matmul(S,torch.matmul(transform_matrix,torch.linalg.inv(T)))
    # grid_trans = torch.linalg.inv(M_torch)[:,0:3,:]

    if if_center_align:
        grid_trans[:,:3,-1] = 0

    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1], output_size[2])))
    img = F.grid_sample(img, grid)
    if abs_value:
        img = torch.where(img>0.5, torch.ones_like(img), torch.zeros_like(img))

    return img



def feature_align(meshes: Meshes, img3D: torch.Tensor):
    """
    Feature align the img3D to the meshes
    """
    # Get the node location
    B, C, D, H, W = img3D.shape

    node_loc = meshes.verts_padded().unsqueeze(1).unsqueeze(1) # [B, 1, 1, N, 3]

    assert node_loc.shape[0] == B

    feature_align= F.grid_sample(img3D, node_loc, mode='bilinear')

   

    return feature_align.squeeze(-2).squeeze(-2).transpose(-1,-2)


def feature_align_with_turbulence(meshes: Meshes, img3D: torch.Tensor, length: int=2, resolution: int=128):
    """
    Feature align the img3D to the meshes with turbulence along the normal direction
    and return the padded feature align at the interior and exterior of the mesh 
    return: feature_align_dict: {'on_surface': feature_align_surface, 'interior': feature_align_interior, 'exterior': feature_align_exterior}
    """

    B, C, D, H, W = img3D.shape
    # Get the node location

    node_loc  = meshes.verts_padded().unsqueeze(1).unsqueeze(1) # [B, 1, 1, N, 3]

    assert node_loc.shape[0] == B
    # Get the node normal
    node_normal = meshes.verts_normals_padded().unsqueeze(1).unsqueeze(1) # [B, 1, 1, N, 3]

    interior_loc = node_loc - node_normal/resolution*2*length

    exterior_loc = node_loc + node_normal/resolution*2*length


    feature_align_surface = F.grid_sample(img3D, node_loc, mode='bilinear')

    feature_align_interior = F.grid_sample(img3D, interior_loc, mode='bilinear')

    feature_align_exterior = F.grid_sample(img3D, exterior_loc, mode='bilinear')

    feature_align_dict = {'on_surface': feature_align_surface.squeeze(-2).squeeze(-2).transpose(-1,-2),
                            'interior': feature_align_interior.squeeze(-2).squeeze(-2).transpose(-1,-2),
                            'exterior': feature_align_exterior.squeeze(-2).squeeze(-2).transpose(-1,-2)}

    return feature_align_dict

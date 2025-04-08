import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch3d.transforms import axis_angle_to_matrix

from pytorch3d.ops import cubify, taubin_smoothing

import numpy as np
import nibabel as nib

from einops import rearrange, repeat


def img_standardize(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std

def img_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)

def load_nib_image(path, omit_tranlation = False):
    '''
    Load the image from the nii file and return the image, voxel size and affine matrix
    Args:
    path: the path of the nii file
    omit_tranlation: if True, the translation part of the affine matrix will be set to 0
    '''
    nib_image = nib.load(path)
    image = nib_image.get_fdata()
    header = nib_image.header
    voxel_size = header.get_zooms()
    affine = nib_image.affine
    affine_scale = np.sqrt(np.sum(affine[:3,:3]**2,0))
    if np.linalg.norm(affine_scale-voxel_size)>1e-5:
        rescaler = affine_scale.reshape(-1,1)*np.array(voxel_size).reshape(-1,1)
    else:
        rescaler = np.ones((3,1))

    affine_rescale = affine.copy()
    affine_rescale[:3,:] = affine[:3,:]*rescaler.reshape(-1,1)
            
    window_size = np.array(voxel_size)*(np.array(image.shape)-1)

    if omit_tranlation:
        affine_rescale[:3,3] = 0

    out_dict = {'img': image, 'affine': affine_rescale, 'voxel_size': voxel_size, 'window_size': window_size}

    return out_dict


def get_coord_map_3d(image_shape, affine, rescaler = 1/100):
    '''
    Get the coordinate map of the image from the affine (torch manner) matrix
    Args:
        image_shape: the shape of the image D,H,W (Z,Y,X)
        affine: the affine matrix of the image (numpy manner)
        rescale: default 1/100: [-100,100]mm -> [-1,1]

    return: 
        the coordinate map of the image with shape (B,Z,Y,X,3)
    '''
    if isinstance(affine,torch.Tensor):
        device = affine.device
    elif isinstance(affine,np.ndarray):
        device = torch.device('cpu')
        affine = torch.from_numpy(affine).float()
    else:
        raise ValueError('The affine should be either torch.Tensor or np.ndarray')
    
    if len(affine.shape)==2:
        affine = affine.unsqueeze(0)
    B = affine.shape[0]
    
    Z,Y,X = image_shape
    coord_map = torch.meshgrid([torch.arange(0,image_shape[-3+i]) for i in range(3)], indexing = 'ij')
    coord_map = torch.stack(coord_map[::-1],-1).float().to(device)
    ## Normalize the coordinate to [-1,1]
    ### Centralize the coordinate
    coord_map = coord_map - (torch.tensor(image_shape,dtype=torch.float32,device=device)-1).flip(-1)/2.
    
    ### Apply the affine matrix
    coord_map = coord_map.unsqueeze(0).repeat(B,1,1,1,1) # B,Z,Y,X,3

    coord_map = rearrange(coord_map,'b z y x c -> b (z y x) c')

    # trans = affine[:,:3,3]*torch.tensor([[X,Y,Z]],dtype=torch.float32,device=device)
    # trans = affine[:,:3,3].unsqueeze(-2).repeat(1,Z*Y*X,1)

    coord_map = torch.matmul(coord_map,affine[:,:3,:3].transpose(-1,-2))+(affine[...,:3,3]).unsqueeze(-2)

    coord_map = rearrange(coord_map,'b (z y x) c -> b z y x c',z=Z,y=Y,x=X)

    ### Rescale the coordinate
    coord_map = coord_map*rescaler

    return coord_map

def augment_from_affine(image, affine, output_shape, mode = 'bilinear'):
    """
    Augment the image based on the affine matrix
    Args:
        image (torch.Tensor): [B, C, Z, Y, X]
        affine (torch.Tensor): [B, 4, 4]
        output_shape (tuple): (Z_new, Y_new, X_new)
        mode (str, optional): interpolation mode. Defaults to 'bilinear'.
    Returns:
        torch.Tensor: [B, C, Z_new, Y_new, X_new]
    """
    B, C, Z, Y, X = image.shape
    Z_new, Y_new, X_new = output_shape
    new_shape = (B, C, Z_new, Y_new, X_new)
    new_affine_grid = F.affine_grid(affine[...,:3,:], new_shape)
    image = F.grid_sample(image, new_affine_grid, mode=mode)
    return image


def get_coord_map_3d_normalized(image_shape, affine):
    '''
    Get the coordinate map of the image from the affine (torch manner) matrix
    Args:
        image_shape: the shape of the image D,H,W (Z,Y,X)
        affine: the affine matrix of the image (torch manner)
    return: 
        the coordinate map of the image with shape (B,Z,Y,X,3)
    '''
    if isinstance(affine,torch.Tensor):
        device = affine.device
    elif isinstance(affine,np.ndarray):
        device = torch.device('cpu')
        affine = torch.from_numpy(affine).float()
    else:
        raise ValueError('The affine should be either torch.Tensor or np.ndarray')
    
    if len(affine.shape)==2:
        affine = affine.unsqueeze(0)

    B = affine.shape[0]
    
    
    Z,Y,X = image_shape
    # coord_map = torch.meshgrid([torch.arange(0,image_shape[-3+i]) for i in range(3)], indexing = 'ij')
    coord_map = torch.meshgrid([torch.linspace(-1,1,image_shape[i]) for i in range(3)], indexing = 'ij')
    
    coord_map = torch.stack(coord_map[::-1],-1).float().to(device)

    ### Apply the affine matrix
    coord_map = coord_map.unsqueeze(0).repeat(B,1,1,1,1) # B,Z,Y,X,3

    coord_map = rearrange(coord_map,'b z y x c -> b (z y x) c')


    coord_map = torch.matmul(coord_map,affine[:,:3,:3].transpose(-1,-2))+(affine[...,:3,3]).unsqueeze(-2)

    coord_map = rearrange(coord_map,'b (z y x) c -> b z y x c',z=Z,y=Y,x=X)

    return coord_map


def label_to_onehot(label, class_list = [0,1,2,3]):
    '''
    Convert the label to one-hot encoding
    label: the input label with shape (N,)
    n_class: the number of classes
    c_dim: the dimension of the channel
    '''

    label = label.long()
    label_onehot = torch.zeros_like(label).long()
    for i, class_value in enumerate(class_list):
        label_onehot[label == class_value] = i+0.0
    label_onehot = F.one_hot(label, num_classes = len(class_list))
    return label_onehot.float()

def onehot_to_label(label_onehot, class_list = [0,1,2,3]):
    '''
    Convert the one-hot encoding to label
    label_onehot: the input label with shape (N, C)
    n_class: the number
    '''
    label = torch.argmax(label_onehot, dim = -1).long()
    label = torch.tensor(class_list, device = label.device)[label]
    
    return label


def warp_img_torch_3D(img, affine, output_size, mode = 'bilinear', if_center_align=True, if_align_corners = True):
    # please pay exact attention on the order of H and W,
    # and the normalization of the grid in Torch, but not in OpenCV
    device = img.device
    B, C, D, H, W = img.shape
    T = torch.Tensor([[(W-1)/2, 0, 0, 0],
                    [0, (H-1)/2, 0, 0],
                    [0, 0, (D-1)/2, 0],
                    [0, 0, 0, 1]]).to(device).repeat(B,1,1)
    
    S = torch.Tensor([[(output_size[2]-1)/2, 0, 0, 0],
                    [0, (output_size[1]-1)/2, 0, 0],
                    [0, 0, (output_size[0]-1)/2, 0],
                    [0, 0, 0, 1]]).to(device).repeat(B,1,1)
    
    if not if_center_align:
        T[:,0:3,3] = T[:,0:3,:].sum(dim = -1)
        S[:,0:3,3] = S[:,0:3,:].sum(dim = -1)

    if affine.shape[-2] == 3:
        print('affine is 3', affine.shape)
        affine = torch.cat((affine, torch.tensor([[[0,0,0,1]]]).to(device).repeat(affine.shape[0],1,1)), dim = -2)
    

    grid_trans = torch.matmul(torch.matmul(affine,T).inverse(),S)[:,:3,:]                 

    # M_torch = torch.matmul(S,torch.matmul(transform_matrix,torch.linalg.inv(T)))
    # grid_trans = torch.linalg.inv(M_torch)[:,0:3,:]

    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1], output_size[2])), align_corners=if_align_corners)
    img = F.grid_sample(img, grid, align_corners=if_align_corners, mode=mode)

    return img




def warp_img_torch_2D(img, affine, output_size, mode = 'bilinear', if_align_corners = False, if_center_align=True):
    # please pay exact attention on the order of H and W,
    # and the normalization of the grid in Torch, but not in OpenCV
    device = affine.device
    B, C, H, W = img.shape
    
    T = torch.Tensor([[(W-1)/2, 0, 0],
                      [0, (H-1)/2, 0],
                      [0, 0, 1]]).to(device).repeat(B,1,1)
    
    S = torch.Tensor([[(output_size[1]-1)/2, 0, 0],
                    [0, (output_size[0]-1)/2, 0],
                    [0, 0, 1]]).to(device).repeat(B,1,1)
    
    if not if_center_align:
        T[:,0:2,2] = T[:,0:2,:].sum(dim = -1)
        S[:,0:2,2] = S[:,0:2,:].sum(dim = -1)

    
    if affine.shape[-2] == 2:
        affine = torch.cat((affine, torch.tensor([[[0,0,1]]]).to(device).repeat(affine.shape[0],1,1)), dim = -2)

    grid_trans = torch.matmul(torch.matmul(affine,T).inverse(),S)[:,:2,:]


    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1])), align_corners=if_align_corners)
    img = F.grid_sample(img, grid, align_corners=if_align_corners, mode=mode)
    
    return img


def affine_augmentation(raw_tensor, perturb_affine, ouput_shape, mode):
    '''
    raw_tensor: torch.Tensor, BxCxZxYxX or BxCxYxX
    perturb_affine: torch.Tensor, Bx4x4 or Bx3x3
    ouput_shape: tuple, (Z, Y, X) or (Y, X)
    mode: str,
    '''
    if len(raw_tensor.shape) == 4:
        B, C, Y, X = raw_tensor.shape
        dim = 2
        assert perturb_affine.shape[-1] == 3
        assert len(ouput_shape) == 2
        
    if len(raw_tensor.shape) == 5:
        B, C, Z, Y, X = raw_tensor.shape
        assert perturb_affine.shape[-1] == 4
        assert len(ouput_shape) == 3
        dim = 3

    ## random affine transformation (applied only on slices)
    # random rotation
    if dim == 2:
        image_warp = warp_img_torch_2D(raw_tensor, perturb_affine, ouput_shape, mode=mode)
    if dim == 3:
        image_warp = warp_img_torch_3D(raw_tensor, perturb_affine, ouput_shape, mode=mode)

    return image_warp

    
    

def random_affine(B, dim = 2, rot_range = [-np.pi, np.pi], trans_range = [-0.1, 0.1], scale_range = [0.9, 1.2]):
    '''
    B: int, batch size
    scale_range: list, [min, max]
    rot_range: list, [min
    '''
    if dim == 2:
        random_affine_matrix = torch.eye(3).unsqueeze(0).repeat(B,1,1)
        random_theta = torch.rand(B,1)*(rot_range[1]-rot_range[0])+rot_range[0]
        random_rot = torch.cat((torch.cos(random_theta), -torch.sin(random_theta), torch.sin(random_theta), torch.cos(random_theta)), dim = -1).view(-1,2,2)
        random_affine_matrix[:,:2,:2] = random_rot
        
        random_trans = torch.rand(B,1)*(trans_range[1]-trans_range[0])+trans_range[0]
        random_affine_matrix[:,:2,-1] = random_trans
        random_scale = torch.rand(B,1)*(scale_range[1]-scale_range[0])+scale_range[0]
        random_scale = random_scale.view(-1,1,1)
        random_affine_matrix[:,:2,:2] = random_affine_matrix[:,:2,:2] * random_scale

    if dim == 3:
        random_affine_matrix = torch.eye(4).unsqueeze(0).repeat(B,1,1)
        random_rotvec = torch.rand(B,3)*(rot_range[1]-rot_range[0])+rot_range[0]
        random_affine_matrix[...,:3,:3] = axis_angle_to_matrix(random_rotvec.unsqueeze(0))
        random_trans = torch.rand(B,3)*(trans_range[1]-trans_range[0])+trans_range[0]
        random_affine_matrix[:,:3,-1] = random_trans
        random_scale = torch.rand(B,1)*(scale_range[1]-scale_range[0])+scale_range[0]
        random_scale = random_scale.view(-1,1,1)
        random_affine_matrix[:,:3,:3] = random_affine_matrix[:,:3,:3] * random_scale

    return random_affine_matrix



def voxel_to_mesh(voxel, affine, threshold):
    Z, Y, X = voxel.shape[-3:]
    mesh = cubify(voxel, threshold)
    mesh_verts = mesh.verts_padded()
    if len(affine.shape) == 2:
        affine = affine.unsqueeze(0)
    mesh_verts[..., 0] = (mesh_verts[..., 0])/2 * (X-1)
    mesh_verts[..., 1] = (mesh_verts[..., 1])/2 * (Y-1)
    mesh_verts[..., 2] = (mesh_verts[..., 2])/2 * (Z-1)
    mesh_verts = mesh_verts.matmul(affine[..., :3, :3].transpose(-1, -2)) + affine[..., :3, 3]
    mesh = mesh.update_padded(mesh_verts)
    return mesh



class GaussianBlur3D(nn.Module):
    def __init__(self):
        super(GaussianBlur3D, self).__init__()

    def forward(self, x, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd.")

        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(x.device)

        # Apply the convolution
        padding = kernel_size // 2
        return F.conv3d(x, kernel, padding=padding)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        coords = coords.expand(kernel_size, kernel_size, kernel_size)
        g = torch.exp(-(coords**2 + coords.transpose(0,1)**2 + coords.transpose(0,2)**2) / (2*sigma**2))
        return g / g.sum()
    
    
def affine_np2torch(affine_np, img_size_np, rescalar = 1/100, center_aligned = True):
    '''
    convert affine matrix from numpy manner to torch manner
    affine_np: [4, 4] original affine matrix read from the medical image file
    img_size: [3] the size of the image, [x, y, z]
    rescalar: [1] the rescalar of the image, default is 1/100mm, [-100mm, 100mm] -> [-1, 1]
    center_aligned: [bool] whether the image is center aligned, default is True
    '''
    if isinstance(affine_np, np.ndarray):
        affine_torch = torch.from_numpy(affine_np).float()
    else:
        affine_torch = affine_np.float()
 

    affine_torch2np = torch.diag(torch.tensor([img_size_np[0]-1., img_size_np[1]-1., img_size_np[2]-1., 1.]))/2.
    affine_torch2np = affine_torch2np.to(affine_torch.device)
    if center_aligned:
        affine_torch[:3, 3] = 0 # set the translation to 0
    else:
        affine_torch2np[:3, 3] = torch.tensor([img_size_np[0]-1., img_size_np[1]-1., img_size_np[2]-1.])/2.
    affine_torch = affine_torch@affine_torch2np
    affine_torch[:3,:] = affine_torch[:3,:]*rescalar
    affine_torch[3,:] = torch.tensor([0,0,0,1]).to(affine_torch.device)
    return affine_torch



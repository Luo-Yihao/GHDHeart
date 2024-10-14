import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from pytorch3d.structures import Meshes
from pytorch3d.ops import mesh_face_areas_normals
from pytorch3d.loss.point_mesh_distance import point_face_distance
from einops import rearrange, einsum, repeat

import numpy as np

from pytorch3d import _C

def faces_angle(meshs: Meshes)->torch.Tensor:
    """
    Compute the angle of each face in a mesh
    Args:
        meshs: Meshes object
    Returns:
        angles: Tensor of shape (N,3) where N is the number of faces
    """
    Face_coord = meshs.verts_packed()[meshs.faces_packed()]
    A = Face_coord[:,1,:] - Face_coord[:,0,:]
    B = Face_coord[:,2,:] - Face_coord[:,1,:]
    C = Face_coord[:,0,:] - Face_coord[:,2,:]
    
    angle_0 = torch.arccos(-torch.sum(C*A,dim=1)/(1e-8+(torch.norm(C,dim=1)*torch.norm(A,dim=1))))
    

    angle_1 = torch.arccos(-torch.sum(A*B,dim=1)/(1e-8+(torch.norm(A,dim=1)*torch.norm(B,dim=1))))
    angle_2 = torch.arccos(-torch.sum(B*C,dim=1)/(1e-8+(torch.norm(B,dim=1)*torch.norm(C,dim=1))))
    angles = torch.stack([angle_0,angle_1,angle_2],dim=1)
    return angles

def dual_area_weights_faces(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area weights of 3 vertices of each triangles in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_weight: Tensor of shape (N,3) where N is the number of triangles
        the dual area of a vertices in a triangles is defined as the area of the sub-quadrilateral divided by three perpendicular bisectors
    """
    angles = faces_angle(Surfaces)
    sin2angle = torch.sin(2*angles)
    dual_area_weight = sin2angle/(torch.sum(sin2angle,dim=-1,keepdim=True)+1e-8)
    dual_area_weight = (dual_area_weight[:,[2,0,1]]+dual_area_weight[:,[1,2,0]])/2
    
    
    return dual_area_weight


def dual_area_vertex(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the dual area of each vertices in a mesh
    Args:
        Surfaces: Meshes object
    Returns:
        dual_area_per_vertex: Tensor of shape (N,1) where N is the number of vertices
        the dual area of a vertices is defined as the sum of the dual area of the triangles that contains this vertices
    """
    dual_weights = dual_area_weights_faces(Surfaces)
    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    face2vertex_index = Surfaces.faces_packed().view(-1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum', dim_size=Surfaces.verts_packed().shape[0])
    
    return dual_area_per_vertex.view(-1,1)


def gaussian_curvature(Surfaces: Meshes,return_topology=False)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = faces_angle(Surfaces)

    dual_weights = dual_area_weights_faces(Surfaces)

    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    dual_area_per_vertex = scatter(dual_areas.view(-1), face2vertex_index, reduce='sum')

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')

    curvature = (2*torch.pi - angle_sum_per_vertex)/(dual_area_per_vertex+1e-8)

    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    if return_topology:
        Euler_chara = (curvature*dual_area_per_vertex).sum()/2/torch.pi
        return curvature, Euler_chara
    return curvature

def gaussian_curvature_density(Surfaces: Meshes)->torch.Tensor:
    """
    Compute the gaussian curvature of each vertices in a mesh by local Gauss-Bonnet theorem
    Args:
        Surfaces: Meshes object
        return_topology: bool, if True, return the Euler characteristic and genus of the mesh
    Returns:
        gaussian_curvature: Tensor of shape (N,1) where N is the number of vertices
        the gaussian curvature of a vertices is defined as the sum of the angles of the triangles that contains this vertices minus 2*pi and divided by the dual area of this vertices
    """

    face2vertex_index = Surfaces.faces_packed().view(-1)

    angle_face = faces_angle(Surfaces)

    dual_weights = dual_area_weights_faces(Surfaces)

    dual_areas = dual_weights*Surfaces.faces_areas_packed().view(-1,1)

    angle_sum_per_vertex = scatter(angle_face.view(-1), face2vertex_index, reduce='sum')

    curvature_density = (2*torch.pi - angle_sum_per_vertex)

    # Euler_chara = torch.sparse.mm(vertices_to_meshid.float().T,(2*torch.pi - sum_angle_for_vertices).T).T/torch.pi/2
    # Euler_chara = torch.round(Euler_chara)
    # print('Euler_characteristic:',Euler_chara)
    # Genus = (2-Euler_chara)/2
    #print('Genus:',Genus)
    return curvature_density

def Average_from_verts_to_face(Surfaces: Meshes, feature_verts: torch.Tensor)->torch.Tensor:
    """
    Compute the average of feature vectors defined on vertices to faces by dual area weights
    Args:
        Surfaces: Meshes object
        feature_verts: Tensor of shape (N,C) where N is the number of vertices, C is the number of feature channels
    Returns:
        vect_faces: Tensor of shape (F,C) where F is the number of faces
    """
    assert feature_verts.shape[0] == Surfaces.verts_packed().shape[0]

    dual_weight = dual_area_weights_faces(Surfaces).view(-1,3,1)

    feature_faces = feature_verts[Surfaces.faces_packed(),:]
    
    wg = dual_weight*feature_faces
    return wg.sum(dim=-2)

### winding number

def Electric_strength(q, p):
    """
    q: (M, 3) - charge position
    p: (N, 3) - field position
    """
    assert q.shape[-1] == 3 and len(q.shape) == 2, "q should be (M, 3)"
    assert p.shape[-1] == 3 and len(p.shape) == 2, "p should be (N, 3)"
    q = q.unsqueeze(1).repeat(1, p.shape[0], 1)
    p = p.unsqueeze(0)
    return (p-q)/(torch.norm(p-q, dim=-1, keepdim=True)**3+1e-8)



def Winding_Occupancy(mesh_tem: Meshes, points: torch.Tensor, max_v_per_call=2000):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem: the reference mesh
    points: the points to be evaluated Nx3
    """
    dual_areas = dual_area_vertex(mesh_tem)


    normals_areaic = mesh_tem.verts_normals_packed() * dual_areas.view(-1,1)

    winding_field = torch.zeros(points.shape[0], device=points.device)

    for i in range(0, normals_areaic.shape[0], max_v_per_call):
        vert_elefields_temp = Electric_strength(points, mesh_tem.verts_packed()[i:i+max_v_per_call])
        winding_field += torch.einsum('m n c, n c -> m', vert_elefields_temp, normals_areaic[i:i+max_v_per_call])


    winding_field = winding_field/4/np.pi

    return winding_field


def Winding_Occupancy_Face(mesh_tem: Meshes, points: torch.Tensor, max_v_per_call=2000):
    """
    Involving the winding number to evaluate the occupancy of the points relative to the mesh
    mesh_tem: the reference mesh
    points: the points to be evaluated Nx3
    """
    areaic,normals = mesh_face_areas_normals(mesh_tem)

    normals_areaic = normals * areaic.view(-1,1)

    face_centers = mesh_tem.verts_packed()[mesh_tem.faces_packed()].view(-1,3,3).mean(dim=-2)


    winding_field = torch.zeros(points.shape[0], device=points.device)

    for i in range(0, normals_areaic.shape[0], max_v_per_call):
        face_elefields_temp = Electric_strength(points, face_centers[i:i+max_v_per_call])
        winding_field += torch.einsum('m n c, n c -> m', face_elefields_temp, normals_areaic[i:i+max_v_per_call])

    winding_field = winding_field/4/np.pi

    return winding_field





class Differentiable_Voxelizer(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='vertex'):
        super(Differentiable_Voxelizer, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.Winding_Occupancy = Winding_Occupancy_Face
        elif integrate_method == 'vertex':
            self.Winding_Occupancy = Winding_Occupancy

    def forward(self, mesh_src: Meshes, output_resolution=256, max_v_per_call=2000, if_binary=False):
        """
        mesh_src: the source mesh to be voxelized (should be rescaled into the normalized coordinates [-1,1])
        return_type: the type of the return
        """

        # random sampling in bounding box
        
        resolution = self.bbox_density
        bbox = mesh_src.get_bounding_boxes()[0]
        
        assert torch.abs(bbox.max())<=1 and torch.abs(bbox.min())<=1, "The bounding box should be normalized into [-1,1]"

        # grid sampling in bounding box
        bbox_length = (bbox[:, 1] - bbox[:, 0])
        step_lengths = bbox_length.max() / resolution
        step = (bbox_length / step_lengths).int() + 1

        x = torch.linspace(bbox[0, 0], bbox[0, 1], steps=step[0], device=mesh_src.device)
        y = torch.linspace(bbox[1, 0], bbox[1, 1], steps=step[1], device=mesh_src.device)
        z = torch.linspace(bbox[2, 0], bbox[2, 1], steps=step[2], device=mesh_src.device)

        

        x_index, y_index, z_index = torch.meshgrid(x, y, z)

        slice_length_ranking, slice_direction_ranking = torch.sort(step, descending=False)

        # change the order of the coordinates for the acceleration 
        slice_direction_ranking_reverse = torch.argsort(slice_direction_ranking,descending=False)


        coordinates = torch.stack([x_index, y_index, z_index], dim=-1)


        coordinates = coordinates.permute(slice_direction_ranking.tolist() + [3])


        coordinates = rearrange(coordinates, 'x y z c -> x (y z) c', c = 3, x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])
        occupency_fields = []
        for i in range(0, coordinates.shape[0]):
            tem_charge = coordinates[i]

            occupency_temp = self.Winding_Occupancy(mesh_src, tem_charge, max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

            occupency_fields.append(occupency_temp)

        occupency_fields = torch.stack(occupency_fields, dim=0)

        # embedding the bounding box into the whole space
        resolution_whole = output_resolution
        bbox_index = (bbox +1)*resolution_whole//2
        X_b, Y_b, Z_b = bbox_index.int().tolist()
        whole_image = torch.zeros(resolution_whole, resolution_whole, resolution_whole, device=mesh_src.device)

        bbox_transformed = rearrange(occupency_fields, 'x (y z) -> x y z', x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])

        bbox_transformed = bbox_transformed.permute(slice_direction_ranking_reverse.tolist()).unsqueeze(0).unsqueeze(0)
        # print(bbox_transformed.shape)
        # print((X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1))
        bbox_transformed = F.interpolate(bbox_transformed, size=(X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1), mode='trilinear')
        bbox_transformed = bbox_transformed.squeeze(0).squeeze(0)

        whole_image[X_b[0]:X_b[1]+1, Y_b[0]:Y_b[1]+1, Z_b[0]:Z_b[1]+1] = bbox_transformed

        whole_image = (whole_image.permute(2, 1, 0)).unsqueeze(0)

        return whole_image
    

def differentiable_sdf(mesh: Meshes, query_points: torch.Tensor, max_query_point_batch_size=10000, binary_style='tanh'):
    """
    Compute the signed distance field of the mesh
    Args:
        mesh: Meshes object
        query_points: Tensor of shape (N,3) where N is the number of query points
    Returns:
        sdf: Tensor of shape (N,1) where N is the number of query points
    """

 
    for i in range(0, query_points.shape[0], max_query_point_batch_size):

        points = query_points[i:i+max_query_point_batch_size]

        dist, _ = _C.point_face_dist_forward(points.view(-1, 3),
                                torch.tensor([0], device=mesh.device),
                                mesh.verts_packed()[mesh.faces_packed(),:],
                                torch.tensor([0], device=mesh.device),
                                points.shape[0], 1e-5)
        dist = dist.view(-1, 1)
            

        occupancy = Winding_Occupancy(mesh, points, max_v_per_call=2000).view(-1,1)
        # binary occupancy
        if binary_style == 'tanh':
            occupancy = -torch.tanh((occupancy-0.5)*1000) # [0,1] -> [-1,1]
        if binary_style == 'sign':
            occupancy = (occupancy<0.5).float()
        if i == 0:
            sdf = dist*occupancy
        else:
            sdf = torch.cat([sdf, dist*occupancy])
        
    return sdf





class DifferentiableRasterizer(torch.nn.Module):
    def __init__(self, alpha=1e4):
        '''
        alpha : float to control the elasticity
        '''
        super(DifferentiableRasterizer, self).__init__()
        self.alpha = alpha

    def forward(self, points, mesh):
        '''
        points : Tensor of shape (B,N,3) representing the 3D points
        mesh : Meshes object representing the mesh
        return : prob of shape (B,H,W,D) representing the probability of each voxel being inside the mesh
        '''
        B = points.shape[0]

        first_index = torch.tensor(range(B), device=points.device)*points.shape[1]
        first_index.requires_grad = False





        tri_verts = mesh.verts_packed()[mesh.faces_packed(),:]


        
        face_num_index = mesh.num_faces_per_mesh()

        tri_first_index  = torch.zeros_like(face_num_index)
        first_index.requires_grad = False

        for i in range(1,face_num_index.shape[0]):
            tri_first_index[i] = tri_first_index[i-1] + face_num_index[i-1]
        

        dist = point_face_distance(points.view(-1, 3),
                                        first_index,
                                        tri_verts,
                                        tri_first_index,
                                        points.shape[1], 1e-5)
        dist = dist.view(points.shape[0],points.shape[1])

        prob = torch.exp(-dist*self.alpha)

        return prob
    
    

class ApproxDVS(torch.nn.Module):
    def __init__(self, ops_size=(40,40,40), alpha=0.005, beta=0.01):
        '''
        ops_size : tuple of 3
        alpha : float to control the elasticity
        beta : float to control the stickiness
        '''
        super(ApproxDVS, self).__init__()
        self.ops_size = ops_size
        self.alpha = alpha
        self.beta = beta

    def compte_disp_from_landmark(self, control_points_coords, control_points_values):
        '''
        control_points_coords : N*3 normalized coordinates
        control_points_values : N*3 (usually the displacement vectors)
        '''
        N = control_points_coords.shape[0]
        device = control_points_coords.device

        img_size = self.ops_size

        init_grid = torch.stack(torch.meshgrid([torch.linspace(-1,1,img_size[0]),torch.linspace(-1,1,img_size[1]),torch.linspace(-1,1,img_size[2])]),dim=-1).to(device) # H*W*D*3

        init_grid = init_grid.view(1,img_size[0],img_size[1],img_size[2],3)



        dist = torch.norm(init_grid.repeat(N,1,1,1,1)-control_points_coords.view(N,1,1,1,3),dim=-1,keepdim=True) # N*H*W*D*1

        weights = torch.exp(-dist**2/self.alpha)
        weights = (self.beta+1)*weights/(torch.sum(weights,dim=0,keepdim=True)+self.beta)
        # weights = weights/(torch.sum(weights,dim=0,keepdim=True))


        control_points_values = control_points_values.view(N,1,1,1,3)


        disp = torch.sum(weights*control_points_values,dim=0)


        return disp.permute(3,2,1,0).unsqueeze(0)
    
    def forward(self, orgin_mask, control_points_coords, control_points_offsets):
        '''
        orgin_mask : Tensor of shape (B,C,D,H,W) C usually is 1
        control_points_coords : Tensor of shape (B,N,3) usually normalized coordinates of the vertices 
        control_points_values : Tensor of shape (B,N,3) usually the displacement vectors of the vertices 
        return : warped_mask of shape (B,C,D,H,W) representing the voxelization of the offset mesh 
        '''


        img_size = orgin_mask.shape[2:]

        B = orgin_mask.shape[0]


        inv_dense_flow_list = []   

        for i in range(B):
            inv_dense_flow = self.compte_disp_from_landmark(control_points_coords[i]+control_points_offsets[i], -control_points_offsets[i])

            inv_dense_flow = F.interpolate(inv_dense_flow, size=img_size, mode='trilinear', align_corners=False)

            inv_dense_flow_list.append(inv_dense_flow)

        inv_dense_flow = torch.cat(inv_dense_flow_list,dim=0)

        warped_grid = torch.stack(torch.meshgrid([torch.linspace(-1,1,img_size[0]),torch.linspace(-1,1,img_size[1]),torch.linspace(-1,1,img_size[2])]),dim=0).to(device=mesh_offset.device).unsqueeze(0)


        warped_grid = warped_grid.permute(0, 1, 4, 3, 2) + inv_dense_flow # 1*3*W*H*D ----> 1*3*D*H*W + B*3*D*H*W


        warped_grid = warped_grid.permute(0, 2, 3, 4, 1) # 1*3*D*H*W -> 1*D*H*W*3
        
        return F.grid_sample(orgin_mask, warped_grid, mode='bilinear', align_corners=False)
    
    
class Differentiable_SDF(nn.Module):
    def __init__(self, bbox_density=128, integrate_method='face'):
        super(Differentiable_Voxelizer, self).__init__()
        self.bbox_density = bbox_density
        if integrate_method == 'face':
            self.Winding_Occupancy = Winding_Occupancy_Face
        elif integrate_method == 'vertex':
            self.Winding_Occupancy = Winding_Occupancy

    def forward(self, mesh_src: Meshes, output_resolution=256, max_v_per_call=2000, if_binary=False):
        """
        mesh_src: the source mesh to be voxelized (should be rescaled into the normalized coordinates [-1,1])
        return_type: the type of the return
        """

        # random sampling in bounding box
        
        resolution = self.bbox_density
        bbox = mesh_src.get_bounding_boxes()[0]
        
        assert torch.abs(bbox.max())<=1 and torch.abs(bbox.min())<=1, "The bounding box should be normalized into [-1,1]"

        # grid sampling in bounding box
        bbox_length = (bbox[:, 1] - bbox[:, 0])
        step_lengths = bbox_length.max() / resolution
        step = (bbox_length / step_lengths).int() + 1

        x = torch.linspace(bbox[0, 0], bbox[0, 1], steps=step[0], device=mesh_src.device)
        y = torch.linspace(bbox[1, 0], bbox[1, 1], steps=step[1], device=mesh_src.device)
        z = torch.linspace(bbox[2, 0], bbox[2, 1], steps=step[2], device=mesh_src.device)

        

        x_index, y_index, z_index = torch.meshgrid(x, y, z)

        slice_length_ranking, slice_direction_ranking = torch.sort(step, descending=False)

        # change the order of the coordinates for the acceleration 
        slice_direction_ranking_reverse = torch.argsort(slice_direction_ranking,descending=False)


        coordinates = torch.stack([x_index, y_index, z_index], dim=-1)


        coordinates = coordinates.permute(slice_direction_ranking.tolist() + [3])


        coordinates = rearrange(coordinates, 'x y z c -> x (y z) c', c = 3, x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])
        occupency_fields = []
        for i in range(0, coordinates.shape[0]):
            tem_charge = coordinates[i]

            occupency_temp = self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)

            if if_binary:
                occupency_temp = torch.sigmoid((self.Winding_Occupancy(mesh_src, tem_charge,max_v_per_call=max_v_per_call)-0.5)*100)

            occupency_fields.append(occupency_temp)

        occupency_fields = torch.stack(occupency_fields, dim=0)

        # embedding the bounding box into the whole space
        resolution_whole = output_resolution
        bbox_index = (bbox +1)*resolution_whole//2
        X_b, Y_b, Z_b = bbox_index.int().tolist()
        whole_image = torch.zeros(resolution_whole, resolution_whole, resolution_whole, device=mesh_src.device)

        bbox_transformed = rearrange(occupency_fields, 'x (y z) -> x y z', x = slice_length_ranking[0], y = slice_length_ranking[1], z = slice_length_ranking[2])

        bbox_transformed = bbox_transformed.permute(slice_direction_ranking_reverse.tolist()).unsqueeze(0).unsqueeze(0)
        # print(bbox_transformed.shape)
        # print((X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1))
        bbox_transformed = F.interpolate(bbox_transformed, size=(X_b[1]-X_b[0]+1, Y_b[1]-Y_b[0]+1, Z_b[1]-Z_b[0]+1), mode='trilinear')
        bbox_transformed = bbox_transformed.squeeze(0).squeeze(0)

        whole_image[X_b[0]:X_b[1]+1, Y_b[0]:Y_b[1]+1, Z_b[0]:Z_b[1]+1] = bbox_transformed

        whole_image = (whole_image.permute(2, 1, 0)).unsqueeze(0)

        return whole_image
    

class MeshThickness(nn.Module):
    def __init__(self, r = 0.2, num_bundle_filtered = 10, innerp_threshold = 0.4, num_sel = 10):
        """
        Args:
            r: the weight of the euclidean distance D_b = D_n + r*D_e
            num_bundle_filtered: the number of faces to be filtered by D_b
            innerp_threshold: remove faces of the normal with inner product < 0.7
            num_sel: the number of faces to be selected finally to calculate the min distance viewed as the thickness 
        """

        super(MeshThickness, self).__init__()

        self.r = r # the weight of the euclidean distance D_b = D_n + r*D_e

        self.num_bundle_filtered = num_bundle_filtered # the number of faces to be filtered by D_b

        self.innerp_threshold = innerp_threshold # remove faces of the normal with inner product < 0.7

        self.num_sel = num_sel # the number of faces to be selected finally to calculate the min distance viewed as the thickness 

        
    def forward(self, current_mesh: Meshes):

        pt, pt_normals = current_mesh.verts_padded(), current_mesh.verts_normals_padded()

        # Get the interior normal 
        pt_normals_inv = -pt_normals

        # Get the faces normals and centeroid
        faces_normals = current_mesh.faces_normals_padded()
        faces_centeroid = current_mesh.verts_packed()[current_mesh.faces_packed(),:].mean(dim=-2)


        ## normal distance
        inner_value = pt_normals_inv.matmul(faces_normals.transpose(-2,-1))
        normal_dist = 1. - inner_value


        ## euclidean distance
        ec_dist = torch.cdist(pt[0], faces_centeroid)
        ec_dist = (ec_dist-ec_dist.mean(dim=-1, keepdim=True))
        ec_dist = ec_dist / ec_dist.std(dim=-1, keepdim=True)


        ## normal bundle distance
        distant_ed_with_normal = normal_dist+ec_dist*self.r
        
        ## filter the top closest faces
        face_index = distant_ed_with_normal.argsort(dim=-1, descending=False)[0,:,:self.num_bundle_filtered]
        selected_normals = faces_normals[:,face_index,:] 


        ## remove the wrong directed faces (double check)
        inner_value_selected = (pt_normals_inv.unsqueeze(-2)*selected_normals).sum(dim=-1)
        inner_value_selected = torch.where(inner_value_selected>0.5, inner_value_selected, -torch.ones_like(inner_value_selected))

        ### get the final face selected
        face_reletive_index_1 = inner_value_selected.argsort(dim=-1, descending=True)[0,:,:self.num_sel]
        face_reletive_index_0 = torch.arange(0,pt.shape[1]).unsqueeze(-1).repeat(1,face_reletive_index_1.shape[-1])
        face_index = face_index[face_reletive_index_0, face_reletive_index_1]


        ## Thickness

        face_vert_indx = current_mesh.faces_packed()[face_index,:]
        tri = current_mesh.verts_packed()[face_vert_indx,:]

        tri_normals = current_mesh.faces_normals_packed()[face_index,:]
        tri_normals = tri_normals.view(-1,3)
        tri_centeroid = tri.mean(dim=-2)
        tri_centeroid = tri_centeroid.view(-1,3)
        

        pt_first_idx = torch.arange(pt.shape[1]).view(-1).to(pt.device)
        tris_first_idx = torch.tensor([i*tri.shape[1] for i in range(tri.shape[0])]).to(tri.device)

        dist, indx = _C.point_face_dist_forward(pt.view(-1,3), pt_first_idx, tri.view(-1,3,3), tris_first_idx, 1, 1e-6)
        ## signed 
        dist_v = (pt.view(-1,3) - tri_centeroid[indx,:])
        sign = -(dist_v*tri_normals[indx,:]).sum(dim=-1)

        closed_indx = face_index.view(-1)[indx]
        return dist, dist_v.norm(dim=-1), closed_indx, sign
    


def normalize_mesh(mesh, rescalar=1.1):
    bbox = mesh.get_bounding_boxes()
    center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2
    size = (bbox[:, :, 1] - bbox[:, :, 0]) 
    scale = 2.0 / (torch.max(size)*rescalar+1e-8)

    mesh = mesh.update_padded((mesh.verts_padded()-center)*scale)
    return mesh

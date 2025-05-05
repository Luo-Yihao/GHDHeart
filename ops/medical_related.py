import torch

def get_4chamberview_frame(cav_pts, lv_pts, rv_pts, **kwargs):
    '''find the frame for 4 chamber view, from points smapled from 3 cavity&ventricle.
    cav_pts: torch.tensor, (N, 3)
    lv_pts: torch.tensor, (N, 3)
    rv_pts: torch.tensor, (N, 3)

    return: target_affine: torch.tensor, (4, 4),

    '''

    

    Pt_center = cav_pts.mean(dim=0) # center of the cavity
    Pt_center_lv = lv_pts.mean(dim=0) # center of the lv
    Pt_center_rv = rv_pts.mean(dim=0) # center of the rv

    Cov_cav = torch.matmul((cav_pts-Pt_center).transpose(0,1), cav_pts-Pt_center)/cav_pts.shape[-2]
    
    _, eigvecs = torch.linalg.eigh(Cov_cav+1e-6*torch.eye(3).to(cav_pts.device))

    u2d_axis = eigvecs[:,-1]


    u2d_axis_proj = (lv_pts-Pt_center).matmul(u2d_axis)
    
    u2d_axis_proj_max, u2d_axis_proj_max_idx = u2d_axis_proj.max(dim=0)
    u2d_axis_proj_min, u2d_axis_proj_min_idx = u2d_axis_proj.min(dim=0)

    dist_max = (lv_pts[u2d_axis_proj_max_idx]-Pt_center) - u2d_axis_proj_max*u2d_axis
    dist_min = (lv_pts[u2d_axis_proj_min_idx]-Pt_center) - u2d_axis_proj_min*u2d_axis


    if dist_max.norm() > dist_min.norm():
        u2d_axis = -u2d_axis

    if 'given_u2d_axis' in kwargs:
        if (u2d_axis*kwargs['given_u2d_axis']).sum() < 0:
            u2d_axis = -kwargs['given_u2d_axis']
        else:
            u2d_axis = kwargs['given_u2d_axis']

    u2d_axis = u2d_axis/torch.norm(u2d_axis)


    l2r_axis = Pt_center_rv-Pt_center
    l2r_axis = l2r_axis/torch.norm(l2r_axis)
    b2f_axis = torch.cross(l2r_axis, u2d_axis)
    b2f_axis = b2f_axis/torch.norm(b2f_axis)
    l2r_axis = torch.cross(u2d_axis, b2f_axis)

    target_frame = torch.stack([-b2f_axis, l2r_axis, -u2d_axis], dim=1)

    target_affine = torch.eye(4).to(cav_pts.device)
    target_affine[:3,:3] = target_frame 
    target_affine[:3,3] = Pt_center

    return {'mean_cav':Pt_center, 'mean_lv':Pt_center_lv, 'mean_rv':Pt_center_rv, 
            'b2f_axis':b2f_axis, 'l2r_axis':l2r_axis, 'u2d_axis':u2d_axis,
            'target_affine':target_affine}
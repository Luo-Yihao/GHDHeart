import torch

def get_4chamberview_frame(cav_pts, lv_pts, rv_pts, **kwargs):
    """
    Find the frame for 4-chamber view from points sampled from 3 cavities and ventricles.
    Args:
        cav_pts (torch.Tensor): Tensor of shape (N, 3) representing points sampled from the cavity.
        lv_pts (torch.Tensor): Tensor of shape (N, 3) representing points sampled from the left ventricle (LV).
        rv_pts (torch.Tensor): Tensor of shape (N, 3) representing points sampled from the right ventricle (RV).
        **kwargs: Additional optional arguments.
            - given_u2d_axis (torch.Tensor, optional): Predefined axis for up-to-down direction. If provided, it will be used to adjust the calculated axis.
    Returns:
        dict: A dictionary containing the following keys:
            - 'mean_cav' (torch.Tensor): Center point of the cavity.
            - 'mean_lv' (torch.Tensor): Center point of the left ventricle.
            - 'mean_rv' (torch.Tensor): Center point of the right ventricle.
            - 'b2f_axis' (torch.Tensor): Back-to-front axis of the 4-chamber view.
            - 'l2r_axis' (torch.Tensor): Left-to-right axis of the 4-chamber view.
            - 'u2d_axis' (torch.Tensor): Up-to-down axis of the 4-chamber view.
            - 'target_affine' (torch.Tensor): Transformation matrix of shape (4, 4) representing the target frame.
    Notes:
        - PCA of the cavity points is used to determine the up-to-down (u2d) axis (principal component).
        - The left-to-right (l2r) axis is determined by the vector from the center of the cavity to the center of the right ventricle.
        - The RV's triangular structure is used to determine the long-axis direction by identifying the upper part of the RV that is farthest from the LV's central axis.
    """


    Pt_center = cav_pts.mean(dim=0) # center of the cavity
    Pt_center_lv = lv_pts.mean(dim=0) # center of the lv
    Pt_center_rv = rv_pts.mean(dim=0) # center of the rv

    Cov_cav = torch.matmul((cav_pts-Pt_center).transpose(0,1), cav_pts-Pt_center)/cav_pts.shape[-2]
    
    _, eigvecs = torch.linalg.eigh(Cov_cav+1e-6*torch.eye(3).to(cav_pts.device))

    u2d_axis = eigvecs[:,-1]

    u2d_axis_proj = (rv_pts-Pt_center).matmul(u2d_axis)
    
    u2d_axis_proj_max, u2d_axis_proj_max_idx = u2d_axis_proj.max(dim=0)
    u2d_axis_proj_min, u2d_axis_proj_min_idx = u2d_axis_proj.min(dim=0)

    dist_max = (rv_pts[u2d_axis_proj_max_idx]-Pt_center) - u2d_axis_proj_max*u2d_axis
    dist_min = (rv_pts[u2d_axis_proj_min_idx]-Pt_center) - u2d_axis_proj_min*u2d_axis


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
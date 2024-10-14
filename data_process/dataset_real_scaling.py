import nibabel as nib

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.','..'))

import torch
import numpy as np
import torch.nn.functional as F

from ops.torch_warping import warp_img_torch_3D
from ops.torch_algebra import random_affine_matrix

import json


import numpy as np


# import skimage 

from torchvision import transforms

from torch.utils.data import Dataset



def read_nii_into_world_coord(nii_path = 'Dataset/ACDC/database/training/patient001/patient001_frame12_gt.nii.gz', device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')), mode = 'trilinear'):
    
    nii_data = nib.load(nii_path)
    img = nii_data.get_fdata()
    affine = nii_data.affine
    header = nii_data.header
    voxel_size = header.get_zooms()

    if len(img.shape) == 4:
        img = img[:,:,:,0]
        voxel_size = voxel_size[:-1]
    
    window = np.array(voxel_size)*(np.array(img.shape)-1)


    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    affine = torch.from_numpy(affine).unsqueeze(0).float().to(device)

    out_put_size_ratio = np.max((np.array(img.shape[-3:])-1)/window)

    output_size = window*out_put_size_ratio+1
    output_size = output_size.astype(int)


    warp_img = F.interpolate(img, size = (int(output_size[0]),int(output_size[1]),int(output_size[2])), mode = mode)

    

    
    identity_affine = torch.eye(4).unsqueeze(0).to(device)

    output_width = np.max(warp_img.shape[-3:])

    output_window_size = (output_width-1)/out_put_size_ratio

    output_width = int(output_width)


    warped_img =  warp_img_torch_3D(warp_img, identity_affine, [output_width]*3, if_center_align=True)

    return warped_img, output_window_size



def mask_img3d(img, mask_value_list, eps = 1e-5):

    mask_img = torch.zeros_like(img).bool()
    for mask_value in mask_value_list:
        mask_tmp = torch.where(torch.abs(img-mask_value)<eps, torch.ones_like(img), torch.zeros_like(img)).bool()
        mask_img = mask_img+mask_tmp

    return mask_img


def augmentation(aligned_img, rot_factor = 0.0, trans_factor = 0.0,  scale_facter = 0.0, augmented_affine=None):

    
    device = aligned_img.device


    if augmented_affine == None:
        
        augmented_affine = random_affine_matrix(rot_factor = rot_factor, trans_factor = trans_factor).to(device)

    output_size = np.max(aligned_img.shape[-3:])

    output_size = int((np.sqrt(3)-scale_facter*np.abs(np.random.randn(1)))*output_size)

    warped_img =  warp_img_torch_3D(aligned_img,augmented_affine,[output_size]*3, if_center_align=True)
    
    return warped_img, augmented_affine



def meshgrid_from_slices(img3d, sliceidx, window, rescalar = 200.0):
    """
    Generate meshgrid from slices
    img3d: 3D image tensor [..., X, Y, Z]
    sliceidx: slice index in the range of [0, Z-1]
    window: window size in real world coordinate
    rescalar: the rescalar of the window size to normalize the meshgrid to -1,1 <==> [0, rescalar]

    """

    meshgrid_X, meshgrid_Y = torch.meshgrid([torch.linspace(0,img3d.shape[-3]-1,img3d.shape[-3]),torch.linspace(0,img3d.shape[-2]-1,img3d.shape[-2])])

    meshgrid_x = meshgrid_X/(img3d.shape[-3]-1)*2-1
    meshgrid_x = meshgrid_x*window[0]/rescalar
    meshgrid_y = meshgrid_Y/(img3d.shape[-2]-1)*2-1
    meshgrid_y = meshgrid_y*window[1]/rescalar
    meshgrid_z = torch.ones_like(meshgrid_x)*(sliceidx)/(img3d.shape[-1]-1)*2-1
    meshgrid_z = meshgrid_z*window[2]/rescalar

    return meshgrid_x.to(img3d.device), meshgrid_y.to(img3d.device), meshgrid_z.to(img3d.device)

def point_cloud_extractor(img3d,  label_value_list, window, spacing=200, coordinate_order = 'zyx'):
    """
    Extract point cloud from the 3D image tensor in the real world coordinate
    img3d: 3D image tensor [1, 1, X, Y, Z]
    label_value_list: list of label values
    window: window size in real world coordinate
    spacing: the spacing of renormalization, default is 200 mm [-1,1] <==> [-100,100]
    coordinate_order: the order of the coordinate, default is zyx to adapt Pytorch & Pytorch3D
    """
    device = img3d.device

    point_list = []
    if label_value_list == None:
        point_cloud = torch.where(img3d[0,0,...] != 0)
        point_cloud = torch.stack([point_cloud[0],point_cloud[1],point_cloud[2]],dim = -1)
        # normalize to -1,1 from the original resolution
        point_cloud = point_cloud/(torch.tensor(img3d.shape[-3:]).float().to(device)-1)*2-1
        # scale to the real world size
        point_cloud = point_cloud*torch.tensor(window).float().to(device)/2
        # # renormalize by the 200*200*200 mm3 window
        point_cloud = point_cloud/spacing*2

        if coordinate_order == 'zyx':
            point_cloud = point_cloud[:,[2,1,0]]
        elif coordinate_order != 'xyz':
            raise ValueError('coordinate_order should be zyx or xyz, zyx is the default value to adapt Pytorch3D')
        point_list.append(point_cloud)

    else:
        for label_value in label_value_list:
            point_cloud = torch.where(img3d[0,0,...]==label_value)
            point_cloud = torch.stack([point_cloud[0],point_cloud[1],point_cloud[2]],dim = -1)
            # normalize to -1,1 from the original resolution
            point_cloud = point_cloud/(torch.tensor(img3d.shape[-3:]).float().to(device)-1)*2-1
            # scale to the real world size
            point_cloud = point_cloud*torch.tensor(window).float().to(device)/2
            # # renormalize by the 200*200*200 mm3 window
            point_cloud = point_cloud/spacing*2


            if coordinate_order == 'zyx':
                point_cloud = point_cloud[:,[2,1,0]]
            elif coordinate_order != 'xyz':
                raise ValueError('coordinate_order should be zyx or xyz, zyx is the default value to adapt Pytorch3D')
            point_list.append(point_cloud)

    return point_list



def fromjson_getaffine_perslice(jsonpath):
    with open(jsonpath) as jsfile:
        jsdict = json.load(jsfile)
        return fromjs_getaffine_perslice(jsdict)

def fromjs_getaffine_perslice(jsdict):
    #https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-affine-formula
    IOP = jsdict['00200037']['Value']
    IPP = jsdict['00200032']['Value']
    PS = jsdict['00280030']['Value']
    #print(IOP, IPP, PS)

    F11, F21, F31 = IOP[3:] # [0020,0037]
    F12, F22, F32 = IOP[:3] # [0020,0037]

    dr, dc = PS #[0028,0030]
    Sx, Sy, Sz = IPP #[0020,0032]

    affine_mat = np.array([
        [F11 * dr, F12 * dc, 0, Sx],
        [F21 * dr, F22 * dc, 0, Sy],
        [F31 * dr, F32 * dc, 0, Sz],
        [0, 0, 0, 1]
    ])
    return affine_mat

def get_affine_transorm(image_data, affine_matrix, pixel_val):
    y,x= torch.where(image_data < pixel_val)
    values = image_data[y, x]
    z = torch.zeros_like(x, dtype=torch.float32)  
    coords = torch.stack([x.float(), y.float(), z, torch.ones_like(x, dtype=torch.float32)])
    transformed_coords = torch.matmul(affine_matrix, coords)
    x_t, y_t, z_t = transformed_coords[:3]
    return x_t, y_t, z_t, values


class UKBB_dataset(Dataset):

    def __init__(self, output_size = (128,128,128), dataset_path = './Dataset/UKBBAtlas/', process_device = 'cuda', label_value_list = [[4.,2.],[2.]], eps=0.1, if_augment=False, rot_factor = 0.0, trans_factor = 0.0,  scale_facter = 0.0):
        self.process_device = process_device
        self.dataset_path = dataset_path
        self.label_value = np.array([ 0., 1., 2., 4.])

        self.label_value_list = label_value_list
        self.output_size = output_size
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.scale_facter = scale_facter
        self.if_augment = if_augment
        self.eps = eps


    def __getitem__(self,index):

        aligned_label, window_size = read_nii_into_world_coord(nii_path = self.dataset_path+'Reg_mask_%d.nii.gz'%(index), device = self.process_device, mode = 'trilinear')

        aligned_label_list = []
        for ele in self.label_value_list:
            aligned_label_list.append(mask_img3d(aligned_label, ele, eps=self.eps).float())

        

        aligned_label = torch.cat(aligned_label_list, dim = -4)
        
        if self.if_augment:
            
            aligned_label, augmented_affine = augmentation(aligned_label, rot_factor = self.rot_factor, trans_factor = self.trans_factor, scale_facter = self.scale_facter)

        aligned_label = F.interpolate(aligned_label, size = self.output_size, mode = 'trilinear')


        # separate the label into different channels
        # warped_label_list = [ warped_label[i:i+1,:,:,:] for i in range(len(self.label_value_list))]


        # warped_label_onehot = F.one_hot(warped_label[0,0,...].long(),num_classes= 2).permute(3,0,1,2).float()

        return {'seg_gt':aligned_label[0], 'window_size': window_size}
    

    def __len__(self):
        return 20


class MMWHS_dataset(Dataset):

    def __init__(self, output_size = (128,128,128), dataset_path = './Dataset/MMWHS/',mode='train', modelity = 'ct', process_device = 'cuda', label_value_list = [[205.,600],[205.]], eps=0.1, if_augment=False, rot_factor = 0.0, trans_factor = 0.0,  scale_facter = 0.0):
        self.process_device = process_device
        self.dataset_path = dataset_path+modelity+'_'+mode+'/'
        self.label_value = np.array([  0., 205., 420., 500., 550., 600., 820., 850.])
        self.mode = mode
        self.modelity_mode = modelity+'_'+mode
        self.nii_path = self.dataset_path+self.modelity_mode+'_'
        self.gt_path = self.dataset_path+self.modelity_mode+'_'
        self.label_value_list = label_value_list
        self.mesh_folder = dataset_path+'mesh_gt/'
        self.output_size = output_size
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.scale_facter = scale_facter
        self.if_augment = if_augment
        self.eps = eps


    def __getitem__(self,index):
        if self.modelity_mode == 'ct_train':
            index = index%20
        elif self.modelity_mode == 'ct_test':
            index = index

        aligned_img, window_size = read_nii_into_world_coord(nii_path = self.nii_path+str(1001+index)+'_image.nii.gz', device = self.process_device)

        if self.mode == 'train':
            aligned_label, window_size = read_nii_into_world_coord(nii_path = self.gt_path+str(index+1001)+'_label.nii.gz', device = self.process_device)
        else:
            aligned_label = torch.zeros_like(aligned_img)

        aligned_label_list = []
        for ele in self.label_value_list:
            aligned_label_list.append(mask_img3d(aligned_label, ele, eps=self.eps).float())


        aligned_label = torch.cat(aligned_label_list, dim = -4)

        aligned_cat = torch.cat([aligned_img, aligned_label], dim = -4)

        if self.if_augment:
            
            aligned_cat, augmented_affine = augmentation(aligned_cat, rot_factor = self.rot_factor, trans_factor = self.trans_factor, scale_facter = self.scale_facter)

        aligned_cat = F.interpolate(aligned_cat, size = self.output_size, mode = 'trilinear')


        warped_img = aligned_cat[0,0:1,:,:,:]


        warped_label = aligned_cat[0,1:,:,:,:]



        # separate the label into different channels
        # warped_label_list = [ warped_label[i:i+1,:,:,:] for i in range(len(self.label_value_list))]


        # warped_label_onehot = F.one_hot(warped_label[0,0,...].long(),num_classes= 2).permute(3,0,1,2).float()

        return {'img': warped_img, 'seg_gt':warped_label, 'window_size': window_size}
    

    def __len__(self):
        return 20


class CCT48_dataset(Dataset):

    def __init__(self, output_size = (128,128,128), dataset_path = './Dataset/2013_cct48/', process_device = 'cuda',  label_value_list = [[205.,600],[205.]], eps=1., if_augment=False, rot_factor = 0.0, trans_factor = 0.0,  scale_facter = 0.0):

        self.process_device = process_device
        self.dataset_path = dataset_path
        self.label_value = np.array([  0., 205., 420., 500., 550., 600., 820., 850.])

        self.label_value_list = label_value_list
        self.output_size = output_size
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.scale_facter = scale_facter
        self.if_augment = if_augment
        self.eps = eps


    def __getitem__(self,index):

        with torch.no_grad():

            aligned_label, window_size = read_nii_into_world_coord(nii_path = self.dataset_path+'image%02d_mas.nii.gz'%(index), device = self.process_device, mode = 'trilinear')
        

            aligned_label_list = []
            for ele in self.label_value_list:
                aligned_label_list.append(mask_img3d(aligned_label, ele, eps=self.eps).float())

            

            aligned_label = torch.cat(aligned_label_list, dim = -4)
            
            if self.if_augment:
                
                aligned_label, augmented_affine = augmentation(aligned_label, rot_factor = self.rot_factor, trans_factor = self.trans_factor, scale_facter = self.scale_facter)

            aligned_label = F.interpolate(aligned_label, size = self.output_size, mode = 'trilinear')


            # separate the label into different channels
            # warped_label_list = [ warped_label[i:i+1,:,:,:] for i in range(len(self.label_value_list))]


        # warped_label_onehot = F.one_hot(warped_label[0,0,...].long(),num_classes= 2).permute(3,0,1,2).float()

        return {'seg_gt':aligned_label[0], 'window_size': window_size}
    

    def __len__(self):
        return 48
    

class ACDC_dataset_Simple(Dataset):

    def __init__(self, dataset_path = './Dataset/MMWHS/',mode='train', 
                 process_device = 'cuda:1'):
        self.process_device = process_device
        self.dataset_path = os.path.join(dataset_path,'database/',mode+'ing/')
        self.label_value = np.array([  0., 1., 2., 3.])
        self.mode = mode


    def __getitem__(self,index):
        if self.mode == 'train':
            index = index+1
        elif self.mode == 'test':
            index = index%50+1+100

        ed_img_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame01.nii.gz'
        ed_label_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame01_gt.nii.gz'


        # search the time point with the ES phase around frame > 1 and with label
        for file in os.listdir(self.dataset_path+'patient'+str(index).zfill(3)+'/'):

            if file.endswith('gt.nii.gz'):
                es_index = int(file.split('_')[1][5:])
                if es_index>1:
                    es_img_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame'+str(es_index).zfill(2)+'.nii.gz'
                    es_label_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame'+str(es_index).zfill(2)+'_gt.nii.gz'
                    break

        nib_label_ed = nib.load(ed_label_path)
        nib_img_ed = nib.load(ed_img_path)
        

        img_ed = nib_img_ed.get_fdata()
        header = nib_img_ed.header

        label_ed = nib_label_ed.get_fdata()
        img_es = nib.load(es_img_path).get_fdata()
        label_es = nib.load(es_label_path).get_fdata()

        voxel_size = header.get_zooms()

        if len(img_ed.shape) == 4:
            img_ed = img_ed[:,:,:,0]
            label_ed = label_ed[:,:,:,0]
            img_es = img_es[:,:,:,0]
            label_es = label_es[:,:,:,0]
            voxel_size = voxel_size[:-1]

        window = np.array(voxel_size)*(np.array(img_ed.shape)-1)


        img_ed = torch.from_numpy(img_ed).unsqueeze(0).unsqueeze(0).float().to(self.process_device)
        label_ed = torch.from_numpy(label_ed).unsqueeze(0).unsqueeze(0).float().to(self.process_device)
        img_es = torch.from_numpy(img_es).unsqueeze(0).unsqueeze(0).float().to(self.process_device)
        label_es = torch.from_numpy(label_es).unsqueeze(0).unsqueeze(0).float().to(self.process_device)
        

        return {'img_ed': img_ed, 'seg_gt_ed':label_ed, 'img_es': img_es, 'seg_gt_es':label_es, 'window': window}
    

    def __len__(self):
        return 100
    

class UKBB_dataset_Simple(Dataset):
    def __init__(self, dataset_path='./data/Biobank/nii_week1/', process_device='cuda:1'):
        self.process_device = process_device
        self.dataset_path = dataset_path
        self.json_file_path = os.path.join(dataset_path, 'json')
        self.nii_file_path = os.path.join(dataset_path, 'NII')
        self.label_value = np.array([0., 1., 2., 3.])
        
        # load the data paths
        self.data_info = self._load_data_paths()

    def _load_data_paths(self):
        patients = []
        for patient_id in os.listdir(self.json_file_path):
            patient_json_dir = os.path.join(self.json_file_path, patient_id)
            patient_nii_dir = os.path.join(self.nii_file_path, f"{patient_id}_inst3")

            
            if os.path.exists(patient_json_dir) and os.path.exists(patient_nii_dir):
                nii_file_list = os.listdir(patient_nii_dir)
                data_info = []
                label_files = [f for f in nii_file_list if '_label.nii.gz' in f]
                
                for label_file in label_files:
                    base_filename = label_file.replace('_label.nii.gz', '')
                    img_nii_path = os.path.join(patient_nii_dir, base_filename + '.nii.gz')
                    json_path = os.path.join(patient_json_dir, base_filename + '.json')
                    seg_nii_path = os.path.join(patient_nii_dir, label_file)

                    data_info.append((img_nii_path, seg_nii_path, json_path))
                patients.append(data_info)
        
        return patients

    def __getitem__(self, index):
        timepoint_data = [{'img_data': [], 'seg_data': [], 'affine': [], 'point_cloud': []} for _ in range(50)]
        
        for img_path, seg_path, json_path in self.data_info[index]:
            affine_matrix = fromjson_getaffine_perslice(json_path) 
            img_nii = nib.load(img_path)
            seg_nii = nib.load(seg_path)
            
            for t in range(50):
                img_data = torch.from_numpy(img_nii.dataobj[..., t].astype(np.float32)).to(self.process_device)
                seg_data = torch.from_numpy(seg_nii.dataobj[..., t].astype(np.float32)).to(self.process_device)
                affine_tensor = torch.from_numpy(affine_matrix.astype(np.float32)).to(self.process_device)

                # if 'LAX' in img_path:
                #     seg_data = seg_data.t()

                x,y,z,v = get_affine_transorm(seg_data, affine_tensor, pixel_val=5)

                timepoint_data[t]['img_data'].append(img_data)
                timepoint_data[t]['seg_data'].append(seg_data)
                timepoint_data[t]['affine'].append(affine_tensor)

                points = torch.stack([x, y, z, v], dim=1)
                timepoint_data[t]['point_cloud'].append(points)

        return timepoint_data
    
    def __len__(self):
        return len(self.data_info)
    

class UKBBAtlas_Simple_dataset(Dataset):

    def __init__(self,  dataset_path = './Dataset/UKBBAtlas/', process_device = 'cuda'):
        self.process_device = process_device
        self.dataset_path = dataset_path
        self.label_value = np.array([ 0., 1., 200., 400.])

    def __getitem__(self,index):
        
        img_path = os.path.join(self.dataset_path, 'Template_3d_%d.nii.gz'%index)
        
        label_path = os.path.join(self.dataset_path, 'Reg_mask_%d.nii.gz'%index)


        try:
            nib_img = nib.load(img_path)
        except:
            nib_img = nib.load(label_path)

        nib_label = nib.load(label_path)
        

        img = nib_img.get_fdata()
        header = nib_img.header
        label = nib_label.get_fdata()

        voxel_size = header.get_zooms()


        window = np.array(voxel_size)*(np.array(img.shape)-1)


        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.process_device)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float().to(self.process_device)


        label = torch.where(torch.abs(label-200)<100., 200., label)
        label = torch.where(label>300., 400., label)
        label = torch.where((label>0.5)*(label<100.), 1., label)
        label = torch.where(label<0.5, 0., label)


        return {'img': img, 'seg_gt':label, 'window': window}
    
    def __len__(self):
        return 21

class ACDC_dataset(Dataset):

    def __init__(self, output_size = (128,128,128), dataset_path = './Dataset/MMWHS/',mode='train', process_device = 'cuda:1', label_value_list = [[1.,2.],[1.]], eps=0.1, if_augment=False, rot_factor = 0.0, trans_factor = 0.0,  scale_facter = 0.0):
        self.process_device = process_device
        self.dataset_path = os.path.join(dataset_path,'database/',mode+'ing/')
        self.label_value = np.array([  0., 1., 2., 3.])
        self.mode = mode

        self.label_value_list = label_value_list
        self.mesh_folder = dataset_path+'mesh_gt/'
        self.output_size = output_size
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.scale_facter = scale_facter
        self.if_augment = if_augment
        self.eps = eps


    def __getitem__(self,index):
        if self.mode == 'train':
            index = index+1
        elif self.mode == 'test':
            index = index%50+1+100

        aligned_img_ed, window_size = read_nii_into_world_coord(nii_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame01.nii.gz', device = self.process_device)


        aligned_label_ed, _ = read_nii_into_world_coord(nii_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame01_gt.nii.gz', device = self.process_device)

        # search the time point with the ES phase around frame > 1 and with label

        for file in os.listdir(self.dataset_path+'patient'+str(index).zfill(3)+'/'):

            if file.endswith('gt.nii.gz'):
                es_index = int(file.split('_')[1][5:])
                if es_index>1:
                    aligned_img_es, _ = read_nii_into_world_coord(nii_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame'+str(es_index).zfill(2)+'.nii.gz', device = self.process_device)
                    aligned_label_es, _ = read_nii_into_world_coord(nii_path = self.dataset_path+'patient'+str(index).zfill(3)+'/'+ 'patient'+str(index).zfill(3)+'_frame'+str(es_index).zfill(2)+'_gt.nii.gz', device = self.process_device)
                    break
        

        aligned_label_ed_list = []
        aligned_label_es_list = []
        for ele in self.label_value_list:
            aligned_label_ed_list.append(mask_img3d(aligned_label_ed, ele, eps=self.eps).float())
            aligned_label_es_list.append(mask_img3d(aligned_label_es, ele, eps=self.eps).float())

        aligned_label_ed = torch.cat(aligned_label_ed_list, dim = -4)
        aligned_label_es = torch.cat(aligned_label_es_list, dim = -4)

        aligned_cat = torch.cat([aligned_img_es, aligned_img_ed, aligned_label_es, aligned_label_ed], dim = -4)

        if self.if_augment:
            
            aligned_cat, augmented_affine = augmentation(aligned_cat, rot_factor = self.rot_factor, trans_factor = self.trans_factor, scale_facter = self.scale_facter)
              
        aligned_cat = F.interpolate(aligned_cat, size = self.output_size, mode = 'trilinear')


        warped_img_ed = aligned_cat[0,0:1,:,:,:]

        

        warped_img_es = aligned_cat[0,1:2,:,:,:]

        warped_label = aligned_cat[0,2:,:,:,:].reshape(2,-1,warped_img_es.shape[-3],warped_img_es.shape[-2],warped_img_es.shape[-1])

        warped_label_es = warped_label[0]

        warped_label_ed = warped_label[1]

        # warped_label_onehot = F.one_hot(warped_label[0,0,...].long(),num_classes= 2).permute(3,0,1,2).float()

        return {'img_ed': warped_img_ed, 'seg_gt_ed':warped_label_ed, 'img_es': warped_img_es, 'seg_gt_es':warped_label_es, 'window_size': window_size}
    

    def __len__(self):
        return 100
    



class Mitea_dataset_simple(Dataset):
    def __init__(self, path='/Mitea', device='cuda:0'):
        self.path = path
        self.device = device
        self.label_value = np.array([0., 1., 2., 3.])
        self.img_path = os.path.join(path, 'images')
        self.label_path = os.path.join(path, 'labels')

        self.data_info = self._load_data_paths()

    def _load_data_paths(self):
        scans = []
        for scan in sorted(os.listdir(self.img_path)):
            if scan.endswith('_ED.nii.gz'):
                base_name = scan.split('_ED')[0]  # Remove '_ED.nii.gz'
                if all(os.path.exists(os.path.join(self.img_path, f"{base_name}_{phase}.nii.gz")) and
                       os.path.exists(os.path.join(self.label_path, f"{base_name}_{phase}.nii.gz"))
                       for phase in ['ED', 'ES']):
                    scans.append(base_name)
        return scans

    def __getitem__(self, index):
        base_path = self.data_info[index]
        data = {}
        for phase in ['ED', 'ES']:
            img_path = os.path.join(self.img_path, f"{base_path}_{phase}.nii.gz")
            label_path = os.path.join(self.label_path, f"{base_path}_{phase}.nii.gz")

            nib_img = nib.load(img_path)
            if phase == 'ED':
                header = nib_img.header
                voxel_size = header.get_zooms()

            img = nib_img.get_fdata()
            affine = nib_img.affine
            affine = torch.from_numpy(affine).float().to(self.device)

            label = nib.load(label_path).get_fdata()

            if len(img.shape) == 4:
                img = img[..., 0]
                label = label[..., 0]
                if phase == 'ED':
                    voxel_size = voxel_size[:-1]

            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(self.device)
            label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float().to(self.device)

            data[f'img_{phase.lower()}'] = img
            data[f'seg_gt_{phase.lower()}'] = label
            data['base_path'] = base_path
            data['affine'] = affine

        window = np.array(voxel_size) * (np.array(img.shape[2:]) - 1)
        data['window'] = window

        return data

    def __len__(self):
        return len(self.data_info) 


if __name__ == '__main__':

    mmwhs = MMWHS_dataset(output_size=(128,128,128),dataset_path = '/home/yihao/data/Dataset/MMWHS/',process_device='cpu')
    print(mmwhs[0]['img'].shape, mmwhs[0]['seg_gt'].shape)

    # import matplotlib.pyplot as plt
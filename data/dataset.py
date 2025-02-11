import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import numpy as np
import nibabel as nib

import os

from data import data_utils as dut

from einops import rearrange, repeat

class ACDCDataset_3DLabel(Dataset):

    def __init__(self, dataset_path = './Dataset/ACDC/',mode='train', output_shape = None, center_aligned = True, rescalar = 1/100):
        self.dataset_path = os.path.join(dataset_path,'database/',mode+'ing/')
        self.label_value = np.array([ 0., 1., 2., 3.])
        self.simple_label = np.array([0., 1., 2., 3.]) # [BG, RV, LMYO, LV]
        self.mode = mode
        self.output_shape = output_shape

        self.center_aligned = center_aligned
        self.rescalar = rescalar

        self.patient_label_list = []
        self.patient_image = []
        print('Loading data from: ', self.dataset_path)
        patient_list = os.listdir(self.dataset_path)


        for patient in patient_list:
            file = os.path.join(self.dataset_path, patient)
            if not os.path.isdir(file):
                continue
            for nii in os.listdir(file):
                if nii.endswith('gt.nii.gz'):
                    self.patient_label_list.append(os.path.join(patient,nii))
                    self.patient_image.append(os.path.join(patient,nii.replace('_gt','')))

        print('Total number of loaded data: ', len(self.patient_label_list))

    def __getitem__(self,index):
        # if self.mode == 'train':
        #     index = index+1
        # elif self.mode == 'test':
        #     index = index%50+1+100

        img_path = os.path.join(self.dataset_path,self.patient_image[index])
        img_path_np = img_path.replace('nii.gz','npy')
        if os.path.exists(img_path_np):
            out_dict = np.load(img_path_np, allow_pickle=True).item()
        else:
            out_dict = dut.load_nib_image(img_path, omit_tranlation = True)
            np.save(img_path_np, out_dict)
        
        
        label_path = os.path.join(self.dataset_path,self.patient_label_list[index])
        label_path_np = label_path.replace('nii.gz','npy')
        if os.path.exists(label_path_np):
            label = np.load(label_path_np, allow_pickle=True).item()['img']
        else:
            label_dict = dut.load_nib_image(label_path, omit_tranlation = True)
            np.save(label_path_np, label_dict)
            label = label_dict['img']

        image_torch = torch.from_numpy(out_dict['img']).permute(2,1,0).unsqueeze(0).float()
        label_torch = torch.from_numpy(label).permute(2,1,0).unsqueeze(0).float()
        affine_torch = torch.from_numpy(out_dict['affine']).float()
        window_size_torch = torch.from_numpy(out_dict['window_size']).float()


        ##  [(-1, 1), (-1, 1), (-1, 1), 1] -> [(0, X-1), (0, Y-1), (0, Z-1), 1]
        affine_torch2np = torch.diag(torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.,1.]))/2.
        
        if self.center_aligned:
            affine_torch[:3, 3] = 0 # set the translation to 0
        else:
            affine_torch2np[:3, 3] = torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.])/2.

        affine_torch = affine_torch@affine_torch2np
        affine_torch[:3,:] = affine_torch[:3,:]*self.rescalar
        affine_torch[3,:] = torch.tensor([0,0,0,1])

        if self.output_shape is not None:
            label_torch = F.interpolate(label_torch.unsqueeze(0), size = self.output_shape, mode = 'nearest').squeeze(0) 
            image_torch = F.interpolate(image_torch.unsqueeze(0), size = self.output_shape, mode = 'trilinear').squeeze(0)

        out_dict = {'image': image_torch, 'label': label_torch, 'affine': affine_torch, 'window_size': window_size_torch}

        return out_dict

    def __len__(self):
        return len(self.patient_label_list)
    

    

class MMWHSDataset_3DLabel(Dataset):
    def __init__(self, dataset_path = './Dataset/MMWHS/',mode='train', output_shape = None,
                 modality = 'ct', center_aligned = True, rescalar = 1/100, RML_simple = False):
        super().__init__()
        self.dataset_path = dataset_path
        self.label_value = np.array([  0., 205., 420., 500., 550., 600., 820., 850.])
        self.simple_label = np.array([0., 600., 205., 500.]) # [BG, RV, LMYO, LV]
        self.mode = mode
        self.modality = modality
        self.center_aligned = center_aligned
        self.rescalar = rescalar
        self.output_shape = output_shape
        self.RML_simple = RML_simple
        if self.RML_simple:
            self.label_value = np.array([0., 1., 2., 3.])


    def __getitem__(self,index):

        if self.modality == 'mixed':
            if index%2 == 0:
                modality = 'ct'
            else:
                modality = 'mr'
            index = index%20
            
        else:
            modality = self.modality

        if self.mode == 'train':
            index = index+1
        elif self.mode == 'test':
            index = index+1001
        
        dataset_path = os.path.join(self.dataset_path, modality+'_'+self.mode)
        modality_mode = modality+'_'+self.mode
        self.nii_path = os.path.join(dataset_path, modality_mode+'_')
        self.gt_path = os.path.join(dataset_path, modality_mode+'_')
        
        nii_path_npy = self.nii_path+str(1000+index)+'_image.npy'
        

        if os.path.exists(nii_path_npy):
            out_dict = np.load(nii_path_npy, allow_pickle=True).item()
        else:
            nii_path = self.nii_path+str(1000+index)+'_image.nii.gz'
            out_dict = dut.load_nib_image(nii_path, omit_tranlation = False)
            np.save(nii_path_npy, out_dict)
    
        image_torch = torch.from_numpy(out_dict['img']).permute(2,1,0).unsqueeze(0).float()


        gt_path_npy = self.gt_path+str(1000+index)+'_label.npy'     
        if self.mode == 'train':
            if os.path.exists(gt_path_npy):
                label = np.load(gt_path_npy, allow_pickle=True).item()['img']
            else:
                gt_path = self.gt_path+str(1000+index)+'_label.nii.gz'
                label_dict = dut.load_nib_image(gt_path, omit_tranlation = False)
                np.save(gt_path_npy, label_dict)
                label = label_dict['img']

            label_torch = torch.from_numpy(label).permute(2,1,0).unsqueeze(0).float()

        affine_torch = torch.from_numpy(out_dict['affine']).float()
        window_size_torch = torch.from_numpy(out_dict['window_size']).float()

        ##  [(-1, 1), (-1, 1), (-1, 1), 1] -> [(0, X-1), (0, Y-1), (0, Z-1), 1]
        affine_torch2np = torch.diag(torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.,1.]))/2.
        
        if self.center_aligned:
            affine_torch[:3, 3] = 0 # set the translation to 0
        else:
            affine_torch2np[:3, 3] = torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.])/2.

        affine_torch = affine_torch@affine_torch2np
        affine_torch[:3,:] = affine_torch[:3,:]*self.rescalar
        affine_torch[3,:] = torch.tensor([0,0,0,1])
        if self.mode == 'train':
            if self.RML_simple:
                label_tem = torch.where((label_torch-self.simple_label[1]).abs() < 0.1, self.label_value[1], self.label_value[0])
                label_tem = torch.where((label_torch-self.simple_label[2]).abs() < 0.1, self.label_value[2], label_tem)
                label_tem = torch.where((label_torch-self.simple_label[3]).abs() < 0.1, self.label_value[3], label_tem)
                label_torch = label_tem
                
        if self.output_shape is not None:
            label_torch = F.interpolate(label_torch.unsqueeze(0), size = self.output_shape, mode = 'nearest').squeeze(0) 
            image_torch = F.interpolate(image_torch.unsqueeze(0), size = self.output_shape, mode = 'trilinear').squeeze(0)


        if self.mode == 'train':
            out_dict = {'image': image_torch, 'label': label_torch, 'affine': affine_torch, 'window_size': window_size_torch}
        elif self.mode == 'test':
            out_dict = {'image': image_torch, 'affine': affine_torch, 'window_size': window_size_torch}

        return out_dict
    

    def __len__(self):
        if self.mode == 'train':
            length = 20
        elif self.mode == 'test':
            length = 10
        if self.modality == 'mixed':
            length = length*2
        return length
    

class MMWHSDataset_Augmented(MMWHSDataset_3DLabel):
    
    def __init__(self, dataset_path = './Dataset/MMWHS/', mode='train', modality = 'ct', 
                 output_shape = (128,128,128), aug_dim = 3, RML_simple = False,
                 rotation_range = [-np.pi, np.pi], 
                 scale_range = [1.0, 1.5], trans_range = [-0.1, 0.1], device = 'cpu'):
        super().__init__(dataset_path, mode, modality)
        self.aug_dim = aug_dim
        self.RML_simple = RML_simple
        if RML_simple:
            self.label_value = np.array([  0., 1., 2., 3.])
        self.output_shape = output_shape
        self.rotation_range = np.array(rotation_range)
        self.scale_range = np.array(scale_range)
        self.trans_range = np.array(trans_range)
        self.device = torch.device(device)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

    def __getitem__(self, index):
        example = super().__getitem__(index)
        device = self.device
        image_tem = example['image'].unsqueeze(0)
        label_tem = example['label']
        if self.RML_simple:
            label_tem = torch.where((example['label']-self.simple_label[1]).abs() < 0.1, 1., 0.)
            label_tem = torch.where((example['label']-self.simple_label[2]).abs() < 0.1, 2., label_tem)
            label_tem = torch.where((example['label']-self.simple_label[3]).abs() < 0.1, 3., label_tem)
        label_tem = label_tem.unsqueeze(0)
        affine_tem = example['affine'].unsqueeze(0)
        image_tem = image_tem.to(device)
        label_tem = label_tem.to(device)
        affine_tem = affine_tem.to(device)
        B, C, Z, Y, X = image_tem.shape

        Z_new, Y_new, X_new = self.output_shape

        

        if self.aug_dim == 3:
            ## 3D affine augmentation ##
            rescale = min((Z_new-1)/(Z-1), (Y_new-1)/(Y-1), (X_new-1)/(X-1))

            aug_affine_3d = dut.random_affine(B, dim = 3, rot_range=self.rotation_range,
                                              trans_range=rescale*self.trans_range, scale_range=rescale*self.scale_range)
            aug_affine_3d = aug_affine_3d.to(device)

            image_tem = dut.affine_augmentation(image_tem, aug_affine_3d, (Z_new, Y_new, X_new), mode = 'bilinear')

            affine_aug = affine_tem.matmul(aug_affine_3d.inverse())

            label_tem = dut.affine_augmentation(label_tem, aug_affine_3d, (Z_new, Y_new, X_new), mode = 'nearest')


        elif self.aug_dim == 2:

            assert Z_new<Z, 'Z_new should be smaller than Z for 2D augmentation, but %d>%d'%(Z_new, Z)

            rescale = min((Y_new-1)/(Y-1), (X_new-1)/(X-1))
            
            aug_affine_3d = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)

            aug_affine_2d = dut.random_affine(B, dim = 2, rot_range=self.rotation_range,
                                              trans_range=rescale*self.trans_range, scale_range=rescale*self.scale_range)
            aug_affine_2d = aug_affine_2d.to(device)

            aug_affine_3d[...,:2,:2] = aug_affine_2d[...,:2,:2]
            aug_affine_3d[...,:2,3] = aug_affine_2d[...,:2,2]

            image_tem = rearrange(image_tem, 'b c z y x -> (b z) c y x')

            image_tem = dut.affine_augmentation(image_tem, aug_affine_2d, (Y_new, X_new), mode = 'bilinear')
            affine_aug = affine_tem.matmul(aug_affine_3d.inverse())

            label_tem = rearrange(label_tem, 'b c z y x -> (b z) c y x')
            label_tem = dut.affine_augmentation(label_tem, aug_affine_2d, (Y_new, X_new), mode = 'nearest')

            image_tem = rearrange(image_tem, '(b z) c y x -> b c z y x', b = B)
            label_tem = rearrange(label_tem, '(b z) c y x -> b c z y x', b = B)

            
            select_indx = np.linspace(0, Z-1, Z_new).astype(int).tolist()
            image_tem = image_tem[:,:,select_indx]
            label_tem = label_tem[:,:,select_indx]
            rescale_inv = torch.tensor([1., 1., (Z-1)/(Z_new-1)]).to(device)
            affine_aug[...,:,:3] = affine_aug[...,:,:3].clone()*rescale_inv.view(1,1,3)


        image_tem = image_tem.squeeze(0)
        label_tem = label_tem.squeeze(0)
        affine_aug = affine_aug.squeeze(0)
        return {'image': image_tem, 'label': label_tem, 'affine': affine_aug, 'window_size': example['window_size']}



# class MMWHSDataset_Augmented_Normalized(MMWHSDataset_3DLabel):
#     def __init__(self, dataset_path = './Dataset/MMWHS/', mode='train', modality = 'ct', 
#                  output_shape = (128,128,128), RML_simple = False,
#                  rotation_range = [-np.pi, np.pi], 
#                  scale_range = [1.0, 1.0], trans_range = [-0.2, 0.2], device = 'cpu'):
#         super().__init__(dataset_path, mode, modality)
#         self.RML_simple = RML_simple
#         if RML_simple:
#             self.label_value = np.array([  0., 1., 2., 3.])
#         self.output_shape = output_shape
#         self.rotation_range = np.array(rotation_range)
#         self.scale_range = np.array(scale_range)
#         self.trans_range = np.array(trans_range)
#         self.device = torch.device(device)
#         if not torch.cuda.is_available():
#             self.device = torch.device('cpu')

#     def __getitem__(self, index):
#         example = super().__getitem__(index)
#         device = self.device
#         image_tem = example['image'].unsqueeze(0)
#         label_tem = example['label']
#         if self.RML_simple:
#             label_tem = torch.where((example['label']-self.simple_label[1]).abs() < 0.1, 1., 0.)
#             label_tem = torch.where((example['label']-self.simple_label[2]).abs() < 0.1, 2., label_tem)
#             label_tem = torch.where((example['label']-self.simple_label[3]).abs() < 0.1, 3., label_tem)
#         label_tem = label_tem.unsqueeze(0)
#         affine_tem = example['affine'].unsqueeze(0)
#         image_tem = image_tem.to(device)
#         label_tem = label_tem.to(device)
#         affine_tem = affine_tem.to(device)
        
#         B, C, Z, Y, X = image_tem.shape

#         Z_new, Y_new, X_new = self.output_shape


#         # Get the center of the LV cavity
#         coordinate_map = dut.get_coord_map_3d_normalized(image_tem.shape[-3:], affine_tem)
#         lv_cavity_index =  torch.where(label_tem==2)
#         lv_cavity_center = coordinate_map[lv_cavity_index[0],lv_cavity_index[2],lv_cavity_index[3],lv_cavity_index[4],:].mean(0,keepdim=True)

#         affine_aug  = dut.random_affine(B=1, dim=3, rot_range=self.rotation_range, 
#                                         scale_range=self.scale_range, trans_range=self.trans_range).to(device)

#         affine_aug[...,:3,3] = affine_aug[...,:3,3] + lv_cavity_center


#         # new_affine = affine_aug.inverse()@affine_tem
#         new_affine = affine_tem.inverse()@affine_aug
#         new_shape = (1, 1, Z_new, Y_new, X_new)
#         new_affine_grid = F.affine_grid(new_affine[...,:3,:], new_shape)

#         label_tem = F.grid_sample(label_tem, new_affine_grid, mode='nearest')
#         image_tem = F.grid_sample(image_tem, new_affine_grid)
        

#         image_tem = image_tem.squeeze(0)
#         label_tem = label_tem.squeeze(0)
#         affine_aug = affine_aug.squeeze(0)
#         return {'image': image_tem, 'label': label_tem, 'affine': affine_aug}
 



class ACDCDataset_Augmented(ACDCDataset_3DLabel):
    def __init__(self, dataset_path = './Dataset/MMWHS/', mode='train', 
                 output_shape = (8,128,128),
                 rotation_range = [-np.pi, np.pi], scale_range = [1.0, 1.5], 
                 trans_range = [-0.1, 0.1], device = 'cpu'):
        super().__init__(dataset_path, mode, process_device = device)

        self.label_value = np.array([  0., 1., 2., 3.])
        self.output_shape = output_shape
        self.rotation_range = np.array(rotation_range)
        self.scale_range = np.array(scale_range)
        self.trans_range = np.array(trans_range)
        self.device = torch.device(device)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

    def __getitem__(self, index):
        example = super().__getitem__(index)
        device = self.device
        image_tem = example['image'].unsqueeze(0)
        label_tem = example['label']
        label_tem = torch.where((example['label']-self.label_value[1]).abs() < 0.1, 1., 0.)
        label_tem = torch.where((example['label']-self.label_value[2]).abs() < 0.1, 2., label_tem)
        label_tem = torch.where((example['label']-self.label_value[3]).abs() < 0.1, 3., label_tem)
        label_tem = label_tem.unsqueeze(0)
        affine_tem = example['affine'].unsqueeze(0)
        image_tem = image_tem.to(device)
        label_tem = label_tem.to(device)
        affine_tem = affine_tem.to(device)
        B, C, Z, Y, X = image_tem.shape

        Z_new, Y_new, X_new = self.output_shape

        rescale = min((Z_new-1)/(Z-1), (Y_new-1)/(Y-1), (X_new-1)/(X-1))

        ## 3D affine augmentation ##

        assert Z_new<=Z, 'Z_new should be smaller than Z for 2D augmentation, but %d>%d'%(Z_new, Z)

        rescale = min((Y_new-1)/(Y-1), (X_new-1)/(X-1))
        
        aug_affine_3d = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)

        aug_affine_2d = dut.random_affine(B, dim = 2, rot_range=self.rotation_range,
                                            trans_range=rescale*self.trans_range, scale_range=rescale*self.scale_range)
        aug_affine_2d = aug_affine_2d.to(device)

        aug_affine_3d[...,:2,:2] = aug_affine_2d[...,:2,:2]
        aug_affine_3d[...,:2,3] = aug_affine_2d[...,:2,2]

        image_tem = rearrange(image_tem, 'b c z y x -> (b z) c y x')

        image_tem = dut.affine_augmentation(image_tem, aug_affine_2d, (Y_new, X_new), mode = 'bilinear')
        affine_aug = affine_tem.matmul(aug_affine_3d.inverse())

        label_tem = rearrange(label_tem, 'b c z y x -> (b z) c y x')
        label_tem = dut.affine_augmentation(label_tem, aug_affine_2d, (Y_new, X_new), mode = 'nearest')

        image_tem = rearrange(image_tem, '(b z) c y x -> b c z y x', b = B)
        label_tem = rearrange(label_tem, '(b z) c y x -> b c z y x', b = B)

        
        select_indx = np.linspace(0, Z-1, Z_new).astype(int).tolist()
        image_tem = image_tem[:,:,select_indx]
        label_tem = label_tem[:,:,select_indx]
        rescale_inv = torch.tensor([1., 1., (Z-1)/(Z_new-1)]).to(device)
        affine_aug[...,:,:3] = affine_aug[...,:,:3].clone()*rescale_inv.view(1,1,3)


        image_tem = image_tem.squeeze(0)
        label_tem = label_tem.squeeze(0)
        affine_aug = affine_aug.squeeze(0)
        return {'image': image_tem, 'label': label_tem, 'affine': affine_aug, 'window_size': example['window_size']}
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
    

    

# class MMWHSDataset_3DLabel(Dataset):
#     def __init__(self, dataset_path = './Dataset/MMWHS/',mode='train', output_shape = None,
#                  modality = 'ct', center_aligned = True, rescalar = 1/100, simple_mode = 'full', load_cache = True):
#         '''
#         mode: 'train' or 'test''
#         simple_mode: 'full' or 'simple' or '4chambers'
#         '''
#         super().__init__()
#         self.dataset_path = dataset_path


#         self.label_value = np.array([  0., 205., 420., 500., 550., 600., 820., 850.]) 


#         if output_shape is not None:
#             self.dataset_path_aligned = os.path.join(dataset_path, 'aligned_'+str(output_shape[0])+'_'+str(output_shape[1])+'_'+str(output_shape[2]))
#         else:
#             self.dataset_path_aligned = os.path.join(dataset_path, 'aligned')

#         if simple_mode == 'simple':
#             self.dataset_path_aligned = self.dataset_path_aligned+'_simple'
#             self.simple_label = np.array([0., 600., 205., 500.]) # [BG, RV, LMYO, LV] 
#             self.label_value = np.array([0., 1., 2., 3.])
#         elif simple_mode == '4Chambers':
#             self.dataset_path_aligned = self.dataset_path_aligned+'_4chambers'
#             self.simple_label = np.array([0.,600., 205., 500., 420., 550.]) # [BG, RV, LMYO, LV, LA, RA]
#             self.label_value = np.array([0., 1., 2., 3. ,4., 5., 6.])
#         elif simple_mode == 'full':
#             self.dataset_path_aligned = self.dataset_path_aligned+'_full'
#             self.simple_label = self.label_value 


#         else:
#             raise ValueError('simple_mode should be one of the following: full, simple, 4chambers')


#         self.mode = mode
#         self.modality = modality
#         self.center_aligned = center_aligned
#         self.rescalar = rescalar
#         self.output_shape = output_shape
#         self.simple_mode = simple_mode

        
#         self.load_cache = load_cache
#         if self.load_cache:
#             print('Allow cache loading from: ', self.dataset_path_aligned)
#             if not os.path.exists(self.dataset_path_aligned):
#                 os.makedirs(self.dataset_path_aligned)
#                 print('Cache folder created for ', self.dataset_path_aligned, '!')
#             else:
#                 print('Cache folder exists for ', self.dataset_path_aligned, '!')
#         else:
#             print('Do not allow cache loading! The existing cache will be overwritten!')


#     def __getitem__(self,index):

#         if self.modality == 'mixed':
#             if index%2 == 0:
#                 modality = 'ct'
#                 index = index//2
#             else:
#                 modality = 'mr'
#                 index = (index-1)//2
#             index = index%20
            
#         else:
#             modality = self.modality

#         if self.mode == 'train':
#             index = index+1
#         elif self.mode == 'test':
#             index = index+1001


#         modality_mode = modality+'_'+self.mode

#         dataset_path = os.path.join(self.dataset_path, modality_mode)
        
#         nii_path = os.path.join(dataset_path, modality_mode+'_')
#         gt_path = os.path.join(dataset_path, modality_mode+'_')
        

#         nii_path_aligned_pt = os.path.join(self.dataset_path_aligned, modality_mode+'_'+str(1000+index)+'_image.npy')
#         # gt_path_aligned_npy = os.path.join(dataset_path_aligned, modality_mode+'_'+str(1000+index)+'_label.npy')
        
#         if self.load_cache and os.path.exists(nii_path_aligned_pt):
#             out_dict = torch.load(nii_path_aligned_pt)
#             return out_dict

#         #### Load the image ####

#         nii_path = nii_path+str(1000+index)+'_image.nii.gz'
#         out_dict = dut.load_nib_image(nii_path, omit_tranlation = False)
        
#         image_torch = torch.from_numpy(out_dict['img']).permute(2,1,0).unsqueeze(0).float()


#         if self.mode == 'train':
            
#             gt_path = gt_path+str(1000+index)+'_label.nii.gz'
#             label_dict = dut.load_nib_image(gt_path, omit_tranlation = False)
#             label = label_dict['img']

#             label_torch = torch.from_numpy(label).permute(2,1,0).unsqueeze(0).float()

#         affine_torch = torch.from_numpy(out_dict['affine']).float()
#         window_size_torch = torch.from_numpy(out_dict['window_size']).float()

#         ##  [(-1, 1), (-1, 1), (-1, 1), 1] -> [(0, X-1), (0, Y-1), (0, Z-1), 1]
#         affine_torch2np = torch.diag(torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.,1.]))/2.
        
#         if self.center_aligned:
#             affine_torch[:3, 3] = 0 # set the translation to 0
#         else:
#             affine_torch2np[:3, 3] = torch.tensor([image_torch.shape[-1]-1.,image_torch.shape[-2]-1.,image_torch.shape[-3]-1.])/2.

#         affine_torch = affine_torch@affine_torch2np
#         affine_torch[:3,:] = affine_torch[:3,:]*self.rescalar
#         affine_torch[3,:] = torch.tensor([0,0,0,1])
#         if self.mode == 'train':
            
#             label_tem = torch.zeros_like(label_torch)

#             for i in range(len(self.simple_label)):
#                 label_tem = torch.where((label_torch-self.simple_label[i]).abs() < 1e-3, self.label_value[i], label_tem)
#             label_torch = label_tem

#         if self.output_shape is not None:
#             if self.mode == 'train':
#                 label_torch = dut.label_to_onehot(label_torch, self.label_value)
#                 label_torch = rearrange(label_torch, 'b z y x c -> b c z y x') 
#                 label_torch = F.interpolate(label_torch, size = self.output_shape, mode = 'trilinear')
#                 label_torch = rearrange(label_torch, '1 c z y x -> 1 z y x c')
#                 label_torch = dut.onehot_to_label(label_torch, self.label_value)
            
#             image_torch = F.interpolate(image_torch.unsqueeze(0), size = self.output_shape, mode = 'trilinear').squeeze(0)


#         if self.mode == 'train':
#             out_dict = {'image': image_torch, 'label': label_torch, 'affine': affine_torch, 'window_size': window_size_torch}
#             torch.save(out_dict, nii_path_aligned_pt)
#         elif self.mode == 'test':
#             out_dict = {'image': image_torch, 'affine': affine_torch, 'window_size': window_size_torch}
#             torch.save(out_dict, nii_path_aligned_pt)

#         return out_dict
    

#     def __len__(self):
#         if self.mode == 'train':
#             length = 20
#         elif self.mode == 'test':
#             if self.modality == 'ct':
#                 length = 40

#         if self.modality == 'mixed':
#             length = length*2
#         return length
    
#     def clean_cache(self):
#         import shutil
#         shutil.rmtree(self.dataset_path_aligned)
#         os.makedirs(self.dataset_path_aligned)
#         print('Cache cleaned for ', self.dataset_path_aligned, '!')

#     def clean_all_cache(self):
#         import shutil
#         all_cache = os.listdir(self.dataset_path)
#         for cache in all_cache:
#             if 'aligned' in cache:
#                 shutil.rmtree(os.path.join(self.dataset_path, cache))
#         print('All Cache cleaned!')
#         os.makedirs(self.dataset_path_aligned)

#     def print_cache(self):
#         all_cache = os.listdir(self.dataset_path)
#         for cache in all_cache:
#             if 'aligned' in cache:
#                 print(cache)
            

#     def show_info(self):
#         print(self.simple_mode)
#         if self.simple_mode == 'full':
#             print('Heart Parts: ', ['BG', 'LMYO', 'LA', 'LV', 'RA', 'RV', 'LAA', 'RAA']) #unsure?
#         elif self.simple_mode == 'simple':
#             print('Heart Parts: ', ['BG', 'RV', 'LMYO', 'LV'])
#         elif self.simple_mode == '4Chambers':
#             print('Heart Parts: ', ['BG', 'RV', 'LMYO', 'LV', 'LA', 'RA'])
#         print('Simple label: ', self.simple_label)
#         print('Label value: ', self.label_value)
    

class MMWHSDataset_3DLabel(Dataset):
    def __init__(self, dataset_path = './Dataset/MMWHS/',mode='train', output_shape = None,
                 modality = 'ct', center_aligned = True, rescalar = 1/100, simple_mode = 'full', load_cache = True):
        '''
        mode: 'train' or 'test''
        simple_mode: 'full' or 'simple' or '4chambers'
        '''
        super().__init__()
        self.dataset_path = dataset_path


        self.label_value = np.array([  0., 205., 420., 500., 550., 600., 820., 850.]) 

        self.dataset_path_aligned = os.path.join(dataset_path, 'preprocessed')

        if simple_mode == 'simple':
            self.simple_label = np.array([0., 600., 205., 500.]) # [BG, RV, LMYO, LV] 
            self.label_value = np.array([0., 1., 2., 3.])
        elif simple_mode == '4Chambers':
            self.simple_label = np.array([0.,600., 205., 500., 420., 550.]) # [BG, RV, LMYO, LV, LA, RA]
            self.label_value = np.array([0., 1., 2., 3. ,4., 5., 6.])
        elif simple_mode == 'full':
            self.simple_label = self.label_value 

        else:
            raise ValueError('simple_mode should be one of the following: full, simple, 4chambers')


        self.mode = mode
        self.modality = modality
        self.center_aligned = center_aligned
        self.rescalar = rescalar
        self.output_shape = output_shape
        self.simple_mode = simple_mode

        
        self.load_cache = load_cache
        if self.load_cache:
            print('Allow cache loading from: ', self.dataset_path_aligned)
            if not os.path.exists(self.dataset_path_aligned):
                os.makedirs(self.dataset_path_aligned)
                print('Cache folder created for ', self.dataset_path_aligned, '!')
            else:
                print('Cache folder exists for ', self.dataset_path_aligned, '!')
        else:
            print('Do not allow cache loading! The existing cache will be overwritten!')


    def __getitem__(self,index):

        if self.modality == 'mixed':
            if index%2 == 0:
                modality = 'ct'
                index = index//2
            else:
                modality = 'mr'
                index = (index-1)//2
            index = index%20
            
        else:
            modality = self.modality

        if self.mode == 'train':
            index = index+1
        elif self.mode == 'test':
            index = index+1001


        modality_mode = modality+'_'+self.mode

        dataset_path = os.path.join(self.dataset_path, modality_mode)
        
        nii_path = os.path.join(dataset_path, modality_mode+'_')
        gt_path = os.path.join(dataset_path, modality_mode+'_')
        

        nii_path_aligned_pt = os.path.join(self.dataset_path_aligned, modality_mode+'_'+str(1000+index)+'_image.npy')
        
        if self.load_cache and os.path.exists(nii_path_aligned_pt):
            out_dict = torch.load(nii_path_aligned_pt)
            image_torch = out_dict['image']
            affine_torch = out_dict['affine']
            window_size_torch = out_dict['window_size']
            if self.mode == 'train':
                label_torch = out_dict['label']
            else:
                label_torch = None
        else:
            #### Load the image from the original nii file ####
            nii_path = nii_path+str(1000+index)+'_image.nii.gz'
            out_dict = dut.load_nib_image(nii_path, omit_tranlation = False)
        
            image_torch = torch.from_numpy(out_dict['img']).permute(2,1,0).unsqueeze(0).float()

            X, Y, Z = image_torch.shape[-1], image_torch.shape[-2], image_torch.shape[-3]


            if self.mode == 'train':
                
                gt_path = gt_path+str(1000+index)+'_label.nii.gz'
                label_dict = dut.load_nib_image(gt_path, omit_tranlation = False)

                label_torch = torch.from_numpy(label_dict['img']).permute(2,1,0).unsqueeze(0).float()

                window_size_torch = torch.from_numpy(out_dict['window_size']).float()

                affine_torch = dut.affine_np2torch(out_dict['affine'], (X,Y,Z), self.rescalar).float()

        ## Apply the simple label
        if self.mode == 'train':
            label_tem = torch.zeros_like(label_torch)

            for i in range(len(self.simple_label)):
                label_tem = torch.where((label_torch-self.simple_label[i]).abs() < 1e-3, self.label_value[i], label_tem)
            label_torch = label_tem.float()


        ## Apply the resize
        if self.output_shape is not None:
            if self.mode == 'train':
                label_torch = self.resize(label_torch, self.output_shape, mode = 'label', label_value = self.label_value)
            image_torch = self.resize(image_torch, self.output_shape, mode = 'img')


        if self.mode == 'train':
            out_dict = {'image': image_torch, 'label': label_torch, 'affine': affine_torch, 'window_size': window_size_torch}
            torch.save(out_dict, nii_path_aligned_pt)
        elif self.mode == 'test':
            out_dict = {'image': image_torch, 'affine': affine_torch, 'window_size': window_size_torch}
            torch.save(out_dict, nii_path_aligned_pt)

        return out_dict
    
    def resize(self, image, output_size, mode='img', label_value = None):
        input_shape = image.shape
        if len(input_shape) ==3:
            image = image.unsqueeze(0).float()
        if len(input_shape) == 4:
            image = image.float()
        if len(input_shape) == 5:
            assert image.shape[0] == 1, f'The input image should be of shape (z y x) or (c z y x) or (1 c z y x), but got: {image.shape}'
            image = image.squeeze(0).float()

        if mode == 'img':
            image = F.interpolate(image.unsqueeze(0), size = output_size, mode = 'trilinear').squeeze(0)
        if mode == 'label':
            if label_value is None:
                label_value = torch.arange(0, image.max()+1).to(image.device)
            image = dut.label_to_onehot(image, label_value)
            image = rearrange(image, 'b z y x c -> b c z y x')
            image = F.interpolate(image, size = output_size, mode = 'trilinear')
            image = rearrange(image, '1 c z y x -> 1 z y x c')
            image = dut.onehot_to_label(image, label_value)
        
        if len(input_shape) == 3:
            image = image.squeeze(0)
        if len(input_shape) == 4:
            pass
        if len(input_shape) == 5:
            image = image.unsqueeze(0)
        return image.float()
    

    def __len__(self):
        if self.mode == 'train':
            length = 20
        elif self.mode == 'test':
            if self.modality == 'ct':
                length = 40

        if self.modality == 'mixed':
            length = length*2
        return length
    
    def clean_cache(self):
        import shutil
        shutil.rmtree(self.dataset_path_aligned)
        os.makedirs(self.dataset_path_aligned)
        print('Cache cleaned for ', self.dataset_path_aligned, '!')

    def clean_all_cache(self):
        import shutil
        all_cache = os.listdir(self.dataset_path)
        for cache in all_cache:
            if 'aligned' in cache or 'preprocessed' in cache:
                shutil.rmtree(os.path.join(self.dataset_path, cache))
        print('All Cache cleaned!')
        os.makedirs(self.dataset_path_aligned)

    def print_cache(self):
        all_cache = os.listdir(self.dataset_path)
        for cache in all_cache:
            if 'aligned' in cache or 'preprocessed' in cache:
                print(cache)
            
    def show_info(self):
        print(self.simple_mode)
        if self.simple_mode == 'full':
            print('Heart Parts: ', ['BG', 'LMYO', 'LA', 'LV', 'RA', 'RV', 'LAA', 'RAA']) #unsure?
        elif self.simple_mode == 'simple':
            print('Heart Parts: ', ['BG', 'RV', 'LMYO', 'LV'])
        elif self.simple_mode == '4Chambers':
            print('Heart Parts: ', ['BG', 'RV', 'LMYO', 'LV', 'LA', 'RA'])
        print('Simple label: ', self.simple_label)
        print('Label value: ', self.label_value)




class Example_Simple_dataset(Dataset):

    def __init__(self,  dataset_path = './Dataset/UKBBAtlas/', resize = 1/100,
                 process_device = 'cuda'):
        self.process_device = process_device
        self.dataset_path = dataset_path
        self.resize = resize
        self.label_value = np.array([ 0., 4., 2., 1.]) # 0: background, 4: RV, 2: LMYO, 1: LV_cavity

    def __getitem__(self,index):
        if index == 0:
            label_path = os.path.join(self.dataset_path, 'label_ED.nii.gz')
        elif index == 1:
            label_path = os.path.join(self.dataset_path, 'label_ES.nii.gz')

        nib_label = nib.load(label_path)
        
        

        header = nib_label.header
        label = nib_label.get_fdata()
        affine = nib_label.affine

        label = torch.from_numpy(label).squeeze(-1).float().to(self.process_device).unsqueeze(0)

        X, Y, Z = label.shape[-3:]

        label = label.permute(0, 3, 2, 1)

        voxel_size = header.get_zooms()


        window = (np.array(voxel_size)*(np.array(label.shape)-1))[:-1]


        affine = torch.from_numpy(affine).float().to(self.process_device)
        affine = dut.affine_np2torch(affine, (X, Y, Z), self.resize)
    
        return {'seg_gt':label, 'window': window, 'affine':affine}
    
        
    
    def __len__(self):
        return 2


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
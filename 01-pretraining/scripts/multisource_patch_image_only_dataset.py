# multisource_patch_image_only_dataset.py - Dataset class for loading image patches without text captions

# --- Setup ---

# imports
import glob
import nibabel as nib
import os
import random
import torch
from torch.utils.data import Dataset

from monai.transforms import Compose as MonaiCompose, LoadImaged


# --- MultisourcePatchImageOnlyDataset class ---
class MultisourcePatchImageOnlyDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None, use_sub_patches=False, base_patch_size=96, sub_patch_size=64):

        # ensure that file paths exist
        if not file_paths:
            raise ValueError("No file paths provided to MultisourcePatchImageOnlyDataset.")
        
        self.file_paths = list(file_paths)
        self.transforms = transforms
        self.use_sub_patches = use_sub_patches
        self.base_patch_size = base_patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patches = [] # list to store sub patches if using sub patches

        # split transforms into load and other transforms
        self.full_transforms = self.transforms
        if self.full_transforms is not None and hasattr(self.full_transforms, 'transforms'):

            # expect exactly 2 blocks [load transforms, train/val transforms]
            try:
                load_transforms, train_val_transforms = self.full_transforms.transforms
            except ValueError:
                raise ValueError('Expected transforms=Compose([load_transforms, train_val_transforms]) in DataModule')
            
            # remove LoadImaged from load_transforms if present
            if load_transforms is not None:
                load_wo_loader = [t for t in getattr(load_transforms, 'transforms', []) if not isinstance(t, LoadImaged)]
                self.transforms_no_load = MonaiCompose([MonaiCompose(load_wo_loader), train_val_transforms])
            else:
                self.transforms_no_load = None
        else:
            self.transforms_no_load = None

        # if using sub patches, ensure sub patch size is smaller than base patch size
        if self.use_sub_patches:
            for p in self.file_paths:
                vol = nib.load(p).get_fdata()
                s = self.sub_patch_size

                # extract all possible sub patches from volume
                starts = [
                    (0,0,0), (s//2,0,0), (0,s//2,0), (0,0,s//2),
                    (s//2,s//2,0), (s//2,0,s//2), (0,s//2,s//2), (s//2,s//2,s//2)
                ]

                # add sub patches to list
                for (x,y,z) in random.sample(starts, 2):
                    sub = vol[z:z+s, y:y+s, x:x+s]
                    self.sub_patches.append((sub, p)) # store sub patch with original file path

    # len
    def __len__(self):
        return len(self.sub_patches) if self.use_sub_patches else len(self.file_paths)
    
    # get item
    def __getitem__(self, idx):
        if self.use_sub_patches:
            image_np, path = self.sub_patches[idx]
            data = {'image': image_np, 'path': path}
            return self.transforms_no_load(data) if self.transforms_no_load else data
        
        else:
            path = self.file_paths[idx]
            data = {'image': path, 'path': path}
            return self.full_transforms(data) if self.full_transforms else data









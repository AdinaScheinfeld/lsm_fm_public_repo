# all_datasets_transforms.py - Functions used for augmentations during pretraining and finetuning

# --- Setup ---

# imports

import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
    ToTensord
)

    
# --- Pretraining Transforms ---

# transform to clamp image intensity between 0-1
class ClampIntensityd(MapTransform):
    def __init__(self, keys, minv=0.0, maxv=1.0):
        super().__init__(keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.clip(d[key], self.minv, self.maxv)
        return d


# function to get training transforms
def get_train_transforms():
    return Compose([

        # spatial augmentations
        RandFlipd(keys=['image'], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=['image'], prob=0.2, max_k=3),
        RandAffined(keys=['image'], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), prob=0.2),

        # intensity augmentations
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.02),
        RandGaussianSmoothd(keys=['image'], prob=0.2),
        RandScaleIntensityd(keys=['image'], factors=0.2, prob=0.2),
        RandShiftIntensityd(keys=['image'], offsets=0.2, prob=0.2),
        ClampIntensityd(keys=['image'], minv=0.0, maxv=1.0),

        # convert to tensor (IMPORTANT)
        ToTensord(keys=['image'])
    ])


# function to get validation transforms
def get_val_transforms():
    return Compose([]) # empty transform for consistency in coding (val only requires loading transforms)



# class to only squeeze if last dim is 1
class SqeeezeLastIfSingleton(MapTransform):

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = d[k]

            # handle numpy and torch
            if hasattr(x, 'shape') and x.ndim >= 4 and x.shape[-1] == 1:
                if isinstance(x, np.ndarray):
                    d[k] = x[..., 0]
                elif torch.is_tensor(x):
                    d[k] = x[..., 0]
        
        return d
                

# get loading transforms
def get_load_transforms(target_size=None):

    # list of transforms
    tfs =  [
        LoadImaged(keys=['image']),
        SqeeezeLastIfSingleton(keys=['image']), # remove trailing singleton dimension if present
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'), # move channel dimension to front or add it if missing
        EnsureTyped(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True)
    ]

    # downsampling
    if target_size is not None:
        tfs.append(Resized(
            keys=['image'],
            spatial_size=(target_size, target_size, target_size),
            mode='trilinear',
            align_corners=False
        ))
    tfs.append(ToTensord(keys=['image']))

    # return all transforms
    return Compose(tfs)
    


# --- Finetuning Transforms ---

# function to get train transforms for finetuning
def get_finetune_train_transforms():

    return Compose([

        # scale intensity to normalize
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),

        # spatial augmentations
        RandFlipd(keys=['image', 'label'], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=['image', 'label'], prob=0.2, max_k=3),
        RandAffined(keys=['image', 'label'], 
                    rotate_range=(0.1, 0.1, 0.1), 
                    scale_range=(0.1, 0.1, 0.1), 
                    prob=0.2, 
                    mode=('bilinear', 'nearest'), 
                    padding_mode='border'),

        # intensity augmentations
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.02),
        ClampIntensityd(keys=['image'], minv=0.0, maxv=1.0),

        # convert to tensor
        ToTensord(keys=['image', 'label'])
    ])


# function to get validation transforms for finetuning
def get_finetune_val_transforms():
    return Compose([
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ])


























# data_module_image_text.py - Image and text data module for all datasets

# --- Setup ---

# imports
from __future__ import annotations

import glob
import os
import random
import sys

from monai.transforms import Compose as MonaiCompose

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

# get dataset class from multisource_patch_image_text_dataset.py
from multisource_patch_image_text_dataset import MultisourcePatchImageTextDataset

# get augmentation functions from other files
sys.path.append('../00-helpers/all_datasets_transforms.py')
from all_datasets_transforms import get_train_transforms, get_val_transforms, get_load_transforms


# --- Data Module ---

# datamodule class for all datasets
class AllDatasetsImageTextDataModule(LightningDataModule):

    # init
    def __init__(self, 
                 roots, # mapping of source name to root directory (connection, dev mouse, human2, selma, wu)
                 enable, # which sources to enable, ex: {'wu': True, 'selma': False, ...}
                 prompt_jsons, # list of json files whose key/prompt mappings will be merged inside the dataset
                 batch_size=4, # batch size per device
                 train_frac=0.9, # fraction of volumes used for training (rest used for validation)
                 seed=100, 
                 data_subset_frac=1.0, # global fraction applied after concatenating sources (keeps post source subsample proportions)
                 per_source_max=None, # optional hard cap on number of files per source (applied before global subsample)
                 per_source_frac=None, # optional per source fractional subsample (applied before cap and before global subset)
                 use_sub_patches=False, # if true, dataset will split each 96^3 patch into 2 64^3 sub-patches
                 base_patch_size=96, # base patch size
                 sub_patch_size=64, # sub-patch size if use_sub_patches=True
                 downsample_to=None, # if not None and use_sub_patches=False, downsample to this size instead of cropping
                 num_workers=4, # dataloader num_workers
                 shuffle_within_source=True # shuffle file lists inside each source before applying per source sampling/caps
                 ):
        
        super().__init__()
        self.roots = roots
        self.enable = enable
        self.prompt_jsons = prompt_jsons
        self.batch_size = int(batch_size)
        self.train_frac = float(train_frac)
        self.seed = int(seed)
        self.data_subset_frac = float(data_subset_frac)
        self.per_source_max = per_source_max or {}
        self.per_source_frac = per_source_frac or {}
        self.use_sub_patches = bool(use_sub_patches)
        self.base_patch_size = int(base_patch_size)
        self.sub_patch_size = int(sub_patch_size)
        self.downsample_to = downsample_to
        self.num_workers = int(num_workers)
        self.shuffle_within_source = bool(shuffle_within_source)

        # warn if batch size is too small
        if self.use_sub_patches and self.batch_size < 16:
            print(f'[WARNING] Using sub-patches with small batch size ({self.batch_size}). Consider increasing batch size.', flush=True)

    # *** discovery helpers for each source ***

    def _files_connection(self, root):
        # .../all_allen_connection_projection_patches/<run>/c0|c1|c2/*.nii.gz
        return glob.glob(os.path.join(root, '*', 'c[0-2]', '*.nii.gz'))

    def _files_dev_mouse(self, root):
        # .../all_allen_developing_mouse_patches/*.nii.gz (flat)
        return glob.glob(os.path.join(root, '*.nii.gz'))

    def _files_human2(self, root):
        # .../all_allen_human2_patches/sub-XXX/*.nii.gz
        return glob.glob(os.path.join(root, '*', '*.nii.gz'))

    def _files_selma(self, root):
        # .../all_selma_patches_96/<class>/*.nii.gz
        return glob.glob(os.path.join(root, '*', '*.nii.gz'))

    def _files_wu(self, root):
        # .../all_wu_brain_patches/<vol>/input/*.nii.gz
        return glob.glob(os.path.join(root, '*/input', '*.nii.gz'))
    
    # function to get all files from enabled sources
    def _collect_all_files(self):

        source_to_files = {}

        if self.enable.get('connection', False) and self.roots.get('connection'):
            source_to_files['connection'] = self._files_connection(self.roots['connection'])
        if self.enable.get('dev_mouse', False) and self.roots.get('dev_mouse'):
            source_to_files['dev_mouse'] = self._files_dev_mouse(self.roots['dev_mouse'])
        if self.enable.get('human2', False) and self.roots.get('human2'):
            source_to_files['human2'] = self._files_human2(self.roots['human2'])
        if self.enable.get('selma', False) and self.roots.get('selma'):
            source_to_files['selma'] = self._files_selma(self.roots['selma'])
        if self.enable.get('wu', False) and self.roots.get('wu'):
            source_to_files['wu'] = self._files_wu(self.roots['wu'])

        return source_to_files
    

    # function to subsample and cap if needed
    def _maybe_subsample_and_cap(self, files, src):

        # shuffle
        fs = list(files)

        if self.shuffle_within_source:
            random.shuffle(fs)

        # per source fractional subsample (before global data_subset_frac)
        frac = float(self.per_source_frac.get(src, 1.0))
        if frac < 1.0:
            keep = max(1, int(len(fs) * frac))
            fs = fs[:keep]

        # per source max cap (before global data_subset_frac)
        cap = self.per_source_max.get(src, None)
        if cap is not None and cap > 0:
            fs = fs[:int(cap)]
        
        return fs
    

    # setup
    def setup(self, stage=None):

        # determine actual input size
        if self.use_sub_patches: # crop to specific size
            target_size = int(self.sub_patch_size)
            mode_msg = f'Using {target_size}^3 sub-patches (crops).'
        elif self.downsample_to is not None: # downsample to specific size
            target_size = int(self.downsample_to)
            mode_msg = f'Downsampling full {self.base_patch_size}^3 -> {target_size}^3 (no cropping).'
        else: # no cropping or downsampling
            target_size = int(self.base_patch_size)
            mode_msg = f'Using {target_size}^3 base patches.'
        print(f'[INFO] {mode_msg}', flush=True)

        # collect files per source
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        src2files_raw = self._collect_all_files()
        for k in list(src2files_raw.keys()):
            src2files_raw[k] = [p for p in src2files_raw[k] if p.endswith('.nii.gz')]

        # subsample and cap per source if needed
        src2files = {src: self._maybe_subsample_and_cap(files, src) for src, files in src2files_raw.items()}

        # concatenate and apply a global subset fraction after per source subsampling/caps
        all_files = []
        src_hint_map = {}
        for src, files in src2files.items():
            all_files.extend(files)
            for f in files:
                src_hint_map[f] = src

        # get only enabled sources
        if not all_files:
            enabled = ', '.join([k for k, v in self.enable.items() if v]) or 'none'
            raise FileNotFoundError(f'No files found. Enabled sources: {enabled}.')

        # shuffle all files
        random.shuffle(all_files)
        if self.data_subset_frac < 1.0:
            keep = max(1, int(len(all_files) * self.data_subset_frac))
            all_files = all_files[:keep]
        
        # get train/val split
        split_idx = int(self.train_frac * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f'[INFO] Enabled sources and counts (after per source sampling):', flush=True)
        for src, files in src2files.items():
            print(f' - {src}: {len(files)}', flush=True)
        print(f'[INFO] Combined -> {len(all_files)} files (subset frac={self.data_subset_frac}).', flush=True)
        print(f'[INFO] Train/val split: {len(train_files)} / {len(val_files)}.', flush=True)

        # datasets
        load = get_load_transforms(target_size=target_size)
        train_tf = MonaiCompose([load, get_train_transforms()])
        val_tf = MonaiCompose([load, get_val_transforms()])

        # create train/val datasets
        self.train_ds = MultisourcePatchImageTextDataset(
            file_paths=train_files, 
            transforms=train_tf,
            prompt_jsons=list(self.prompt_jsons), # merged inside dataset
            use_sub_patches=self.use_sub_patches,
            base_patch_size=self.base_patch_size,
            sub_patch_size=self.sub_patch_size,
            source_hint_map={f: src_hint_map[f] for f in train_files} # only include train files
        )

        self.val_ds = MultisourcePatchImageTextDataset(
            file_paths=val_files, 
            transforms=val_tf,
            prompt_jsons=list(self.prompt_jsons), # merged inside dataset
            use_sub_patches=self.use_sub_patches,
            base_patch_size=self.base_patch_size,
            sub_patch_size=self.sub_patch_size,
            source_hint_map={f: src_hint_map[f] for f in val_files} # only include val files
        )

    # train dataloader
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers,
                          persistent_workers=False,
                          pin_memory=True)
    
    # val dataloader
    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          persistent_workers=False,
                          pin_memory=True)


        





        
# multisource_patch_image_text_dataset.py - Dataset class for image+text pretraining

# --- Setup ---

# imports
import json
import nibabel as nib
import os
import random
import re

from monai.transforms import Compose as MonaiCompose, LoadImaged

from torch.utils.data import Dataset


# --- Helper Functions ---

# function to normalize keys
def _normalize_key(s):
    return s.strip().lower().replace(' ', '_').replace('-', '_')


# function to merge multiple prompt dicts
def load_prompt_dicts(json_paths):
    merged = {}
    
    # ignore missing files
    for jp in json_paths:
        if not os.path.exists(jp):
            continue

        print(f"Loading prompt mappings from {jp}...")
        with open(jp, 'r') as f:
            d = json.load(f)
            for k,v in d.items():
                merged[_normalize_key(k)] = v # normalize keys
    return merged


# --- Dataset Class ---

# nifti patch dataset class with text
class MultisourcePatchImageTextDataset(Dataset):

    # precompile regexes for specific sources
    _re_human2_stain = re.compile(r'[_-]stain-([A-Za-z0-9]+)', re.IGNORECASE)
    _re_dev_mouse_label = re.compile(r'^([^_]+)_ps\d+_', re.IGNORECASE)

    # init
    def __init__(self, 
                 file_paths, 
                 transforms=None, 
                 prompt_jsons=None, 
                 use_sub_patches=False, 
                 base_patch_size=96, 
                 sub_patch_size=64,
                 source_hint_map=None):
        
        if not file_paths:
            raise ValueError('MultiSourceNiftiTextPatchDataset requires a non-empty list of file paths')
        self.file_paths = file_paths
        self.transforms = transforms
        self.use_sub_patches = use_sub_patches
        self.base_patch_size = int(base_patch_size)
        self.sub_patch_size = int(sub_patch_size)
        self.source_hint_map = source_hint_map or {}

        # load stain-text mapping from json file
        if not prompt_jsons:
            raise ValueError('Provide prompt_jsons for stain-text mapping')
        self.prompt_map = load_prompt_dicts(prompt_jsons)

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

        # if using sub_patches, split each file path into sub_patches
        if self.use_sub_patches:
            self.sub_patches = []
            for path in self.file_paths:
                vol = self._load_volume(path) # shape: (1, base_patch_size, base_patch_size, base_patch_size)
                sub_patch_list = self._split_into_sub_patches(vol)
                text = self.extract_text(path)
                for sub_patch in sub_patch_list:
                    self.sub_patches.append((sub_patch, text, path)) # store tuple of (sub_patch, text, path) for debugging

    # *** loading/subpatching ***

    # function to load volume from file path
    def _load_volume(self, path):
        vol = nib.load(path).get_fdata() # shape: (D, H, W)
        return vol
    
    # function to split volume into sub_patches
    def _split_into_sub_patches(self, vol):

        # define valid starts
        s = self.sub_patch_size
        starts = [
            (0, 0, 0), (s//2, 0, 0), (0, s//2, 0), (0, 0, s//2),
            (s//2, s//2, 0), (s//2, 0, s//2), (0, s//2, s//2), (s//2, s//2, s//2)
        ]
        
        # select 2 random non-overlapping subpatches
        selected = random.sample(starts, 2)
        sub_patches = []
        for x, y, z in selected:
            x, y, z = int(x), int(y), int(z)
            sub_patches.append(vol[z:z+s, y:y+s, x:x+s]) # shape: (sub_patch_size, sub_patch_size, sub_patch_size)

        return sub_patches
    
    # *** Prompt extraction for each source ***

    # function to extract text prompt from file
    """
    1. Allen connection projection: parent folder c0/c1/c2.
    2. Allen dev mouse: filename prefix before '_psXX_'.
    3. Human2: parse 'stain-XYZ' from filename.
    4. Selma: immediate parent folder class name (ab_plaque, vessel_wga, ...).
    5. Wu: parent volume folder (gparent of file, when parent -- 'input').
    6. Fallback: generic string.
    """
    def extract_text(self, path):

        # get path, parent, and gparent folder names
        p = path
        fname = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p))
        gparent = os.path.basename(os.path.dirname(os.path.dirname(p)))

        # 1. Allen connection projection: parent folder c0/c1/c2
        if parent.lower() in ('c0', 'c1', 'c2'):
            key = _normalize_key(parent)
            if key in self.prompt_map:
                return self.prompt_map[key]
        
        # 2. Allen dev mouse: filename prefix before '_psXX_'
        m2 = self._re_dev_mouse_label.match(fname)
        if m2:
            key = _normalize_key(m2.group(1))
            if key in self.prompt_map:
                return self.prompt_map[key]

        # 3. Human2: parse 'stain-XYZ' from filename
        m = self._re_human2_stain.search(fname)
        if m:
            key = _normalize_key(m.group(1))
            if key in self.prompt_map:
                return self.prompt_map[key]

        # 4. Selma: immediate parent folder class name
        key = _normalize_key(parent)
        if key in self.prompt_map:
            return self.prompt_map[key]
        
        # 5. Wu: parent volume folder (gparent of file, when parent -- 'input')
        if parent.lower() == 'input' and gparent:
            key = _normalize_key(gparent)
            if key in self.prompt_map:
                return self.prompt_map[key]

        # 6. Fallback: substring match anywhere in normalized path against known keys
        lp = _normalize_key(p)
        for k, v in self.prompt_map.items():
            if k in lp:
                return v
            
        # 7. Final fallback
        return 'Unknown microscopy patch with unannotated staining and location.'

    # *** PyTorch Dataset methods ***

    # length
    def __len__(self):

        # if using sub_patches
        if self.use_sub_patches:
            return len(self.sub_patches)
        
        # if not using sub_patches
        return len(self.file_paths)
    
    # getter
    def __getitem__(self, idx):

        # if using sub_patches
        if self.use_sub_patches:

            # image and text
            image_np, text, src_path = self.sub_patches[idx]
            data = {'image': image_np, 'text': text, 'path': src_path} # include path for debugging
            if self.transforms_no_load:
                data = self.transforms_no_load(data)
            return data
        
        # if not using sub_patches
        else:
            # image and text
            path = self.file_paths[idx]
            data = {'image': path, 'text': self.extract_text(path), 'path': path} # include path for debugging
            if self.full_transforms:
                data = self.full_transforms(data)
            return data




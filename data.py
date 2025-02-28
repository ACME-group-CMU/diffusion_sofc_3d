import os
import numpy as np
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import warnings

warnings.filterwarnings("ignore")


## Create Dataset class
class Microstructures(Dataset):

    def __init__(
        self,
        data_path: str,
        condition_path: str,
        img_size: typing.Tuple[int, int, int] = (64, 64, 64),
        length: int = 5,
        apply_symmetry: bool = True,
        indices: typing.Optional[np.ndarray] = None,
        subset: typing.Optional[str] = None,
    ):
        """
        Initialize the Microstructures dataset.
        
        Args:
            data_path (str): Path to the data.
            condition_path (str): Path to the conditional data.
            img_size (tuple): Size of the images.
            length (int): Length of the dataset.
            apply_symmetry (bool): Whether to apply symmetry.
            indices (np.ndarray, optional): Indices for the dataset.
            subset (str, optional): Subset of the data.
        """
        self.data = np.load(data_path)["data"]
        self.cond_data = np.load(condition_path)
        self.total_samples = length
        self.apply_sym = apply_symmetry
        self.img_size = img_size
        self.indices = indices

        if subset is not None:
            self.data = self.apply_subset(subset)
            self.cond_data = self.apply_subset(self.cond_data)

        if self.indices is not None:
            self.total_samples = self.indices.shape[0]

    def apply_subset(self, subset: str) -> np.ndarray:
        """
        Apply subset to the data.
        
        Args:
            subset (str): Subset string.
        
        Returns:
            np.ndarray: Subset of the data.
        """
        def parse_slice(s):
            start, stop = (s.split(":") + [None])[:2]
            start = int(start) if start else None
            stop = int(stop) if stop else None
            return slice(start, stop)

        slices = tuple(parse_slice(s) for s in subset.split(","))
        return self.data[slices]

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return self.total_samples

    def apply_symmetry_transformations(self, subimage: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetry transformations to the subimage.
        
        Args:
            subimage (torch.Tensor): Subimage tensor.
        
        Returns:
            torch.Tensor: Transformed subimage tensor.
        """
        subimage = torch.rot90(subimage, k=np.random.randint(4), dims=(1, 2))
        subimage = torch.rot90(subimage, k=np.random.choice([0, 2]), dims=(0, 1))
        subimage = torch.rot90(subimage, k=np.random.choice([0, 2]), dims=(0, 2))

        if np.random.choice([True, False]):
            subimage = torch.flip(subimage, dims=[0])
        if np.random.choice([True, False]):
            subimage = torch.flip(subimage, dims=[1])
        if np.random.choice([True, False]):
            subimage = torch.flip(subimage, dims=[2])

        return subimage

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item.
        
        Returns:
            tuple: Subimage and volume fractions.
        """
        im = self.data
        im_dims = np.array(self.data.shape)
        subimage_size = np.array(self.img_size)
        idx_max = im_dims - (subimage_size - 1)

        if self.indices is not None:
            idx = self.indices[idx]
        else:
            idx = [np.random.randint(i) for i in idx_max]

        subimage = im[
            idx[0] : (idx[0] + subimage_size[0]),
            idx[1] : (idx[1] + subimage_size[1]),
            idx[2] : (idx[2] + subimage_size[2]),
        ]

        cond_subimage = self.cond_data[
            idx[0] : (idx[0] + subimage_size[0]),
            idx[1] : (idx[1] + subimage_size[1]),
            idx[2] : (idx[2] + subimage_size[2]),
        ]

        subimage = torch.tensor(subimage)

        if self.apply_sym:
            subimage = self.apply_symmetry_transformations(subimage)

        subimage = ((subimage / 255) * 2) - 1
        subimage = torch.unsqueeze(subimage, 0)

        vol_fracs = np.array([(cond_subimage == i).sum() for i in range(1, 4)])
        vol_fracs = vol_fracs / cond_subimage.size
        vol_fracs = torch.tensor(vol_fracs, dtype=subimage.dtype)

        return subimage, vol_fracs

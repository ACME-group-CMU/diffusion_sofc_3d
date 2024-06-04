import os
import numpy as np

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
        data_path,
        img_size=(64, 64, 64),
        length=5,
        apply_symmetry=True,
        indices=None,
        subset=None,
    ):
        """
        Dataset class that will sample a 3D subimage from the data that is a numpy array
        in data_path

        Parameters
        ----------
        data_path : string
                    Should be the path to a 3D image data stored as a numpy ndarray.

                    It will load the file and put search for "data" array as it is by default a
                    numpy compressed file.

        img_size : tuple of ints
            Size of the subvolume to be sampled.

        length : int
                 length of dataset (because of random sampling)

        apply_symmetry : bool
            If set to True, each subimage will be randomy rotated and flipped.

        Returns
        -------
        subimage : ndarray
            The sampled 3D subimage.
        """

        self.data = np.load(data_path)["data"]
        self.total_samples = length
        self.apply_sym = apply_symmetry
        self.img_size = img_size
        self.indices = indices

        if subset is not None:
            self.data = self.apply_subset(subset)

        if self.indices is not None:
            self.total_samples = self.indices.shape[0]

    def apply_subset(self, subset):
        def parse_slice(s):
            start, stop = (s.split(":") + [None])[:2]
            start = int(start) if start else None
            stop = int(stop) if stop else None
            return slice(start, stop)

        slices = tuple(parse_slice(s) for s in subset.split(","))
        return self.data[slices]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

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

        subimage = torch.Tensor(subimage)

        if self.apply_sym:
            subimage = torch.rot90(subimage, k=np.random.randint(4), dims=(1, 2))
            # subimage = np.rot90(subimage, k=np.random.randint(4), axes=(0,1))
            # subimage = np.rot90(subimage, k=np.random.randint(4), axes=(0,2))
            subimage = torch.rot90(subimage, k=np.random.choice([0, 2]), dims=(0, 1))
            subimage = torch.rot90(subimage, k=np.random.choice([0, 2]), dims=(0, 2))

            if np.random.choice([True, False]):
                subimage = torch.flip(subimage, dims=[0])
            if np.random.choice([True, False]):
                subimage = torch.flip(subimage, dims=[1])
            if np.random.choice([True, False]):
                subimage = torch.flip(subimage, dims=[2])

        subimage = ((subimage / 255) * 2) - 1

        subimage = torch.unsqueeze(subimage, 0)

        return subimage

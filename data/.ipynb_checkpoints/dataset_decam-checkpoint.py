import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
from utils.utils_data import distort_images, resize_crop
from utils.utils import PROJECT_ROOT


class DECAMDataset(Dataset):
    """
    DECAM dataset class used for pre-training the encoders for IQA.

    Args:
        root (string): root directory of the dataset
        patch_size (int): size of the patches to extract from the images
        max_distortions (int): maximum number of distortions to apply to the images
        num_levels (int): number of levels of distortion to apply to the images
        pristine_prob (float): probability of not distorting the images

    Returns:
        dictionary with keys:
            img_A_orig (Tensor): first view of the image pair
            img_B_orig (Tensor): second view of the image pair
            img_A_name (string): name of the image of the first view of the image pair
            img_B_name (string): name of the image of the second view of the image pair
            distortion_functions (list): list of the names of the distortion functions applied to the images
            distortion_values (list): list of the values of the distortion functions applied to the images
    """
    def __init__(self,
                 root: str,
                 patch_size: int = 224,
                 max_distortions: int = 4,
                 num_levels: int = 5,
                 pristine_prob: float = 0.05):

        root = Path(root)

        
        filenames_csv_path = "../data/decam_dr10_good_exp.csv"
        exp_df = pd.read_csv(filenames_csv_path, header=None, names=["expnum"])

        # Path to FITS file containing image data
        file_path = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/survey-ccds-decam-dr10.fits.gz"
        image_table = Table.read(file_path)

        self.ref_images = []
        self.hdu_numbers = []

        # Iterate over rows in image_table
        for row in image_table:
            expnum = row['expnum']
            image_path = os.path.join(root, row['image_filename'])
            hdu_number = row['image_hdu']

            if expnum in exp_df['expnum'].values and os.path.exists(image_path):
                if image_path not in self.ref_images:
                    self.ref_images.append(image_path)
                    self.hdu_numbers.append(hdu_number)


        # Convert paths to Path objects
        self.ref_images = [Path(path) for path in self.ref_images]
        self.patch_size = patch_size
        self.max_distortions = max_distortions
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_levels = num_levels
        self.pristine_prob = pristine_prob

        assert 0 <= self.max_distortions <= 7, "The parameter max_distortions must be in the range [0, 7]"
        assert 1 <= self.num_levels <= 5, "The parameter num_levels must be in the range [1, 5]"

    def __getitem__(self, index: int) -> dict:
        img_A_path = self.ref_images[index]
        hdu_number = self.hdu_numbers[index]
        hdul_A = fits.open(img_A_path)
        img_A = hdul_A[hdu_number].data

        # Select another exposure randomly
        other_exp_index = np.random.choice(np.setdiff1d(range(len(self.ref_images)), [index]))
        img_B_path = self.ref_images[other_exp_index]
        hdul_B = fits.open(img_B_path)
        img_B = hdul_B[hdu_number].data
        
        # Resize and crop
        img_A_orig = resize_crop(img_A, self.patch_size)
        img_B_orig = resize_crop(img_B, self.patch_size)

        img_A_orig = transforms.ToTensor()(img_A_orig)
        img_B_orig = transforms.ToTensor()(img_B_orig)

        distort_functions_A = []
        distort_values_A = []
        distort_functions_B = []
        distort_values_B = []

        # Distort images with (1 - self.pristine_prob) probability for image A
        if random.random() > self.pristine_prob and self.max_distortions > 0:
            img_A_orig, distort_functions_A, distort_values_A = distort_images(img_A_orig,
                                                                                 max_distortions=self.max_distortions,
                                                                                 num_levels=self.num_levels)

        # Distort images with (1 - self.pristine_prob) probability for image B
        if random.random() > self.pristine_prob and self.max_distortions > 0:
            img_B_orig, distort_functions_B, distort_values_B = distort_images(img_B_orig,
                                                                                 max_distortions=self.max_distortions,
                                                                                 num_levels=self.num_levels)

        img_A_orig = self.normalize(img_A_orig)
        img_B_orig = self.normalize(img_B_orig)

        # Pad to make the length of distort_functions and distort_values equal for all samples
        distort_functions_A = [f.__name__ for f in distort_functions_A]
        distort_functions_A += [""] * (self.max_distortions - len(distort_functions_A))
        distort_values_A += [torch.inf] * (self.max_distortions - len(distort_values_A))

        distort_functions_B = [f.__name__ for f in distort_functions_B]
        distort_functions_B += [""] * (self.max_distortions - len(distort_functions_B))
        distort_values_B += [torch.inf] * (self.max_distortions - len(distort_values_B))

        return {
            "img_A_orig": img_A_orig,"img_B_orig": img_B_orig,
            "distortion_functions_A": distort_functions_A, "distortion_values_A": distort_values_A,
            "distortion_functions_B": distort_functions_B, "distortion_values_B": distort_values_B
        }

    def __len__(self) -> int:
        return len(self.ref_images)


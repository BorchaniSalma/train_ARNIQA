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
from torchvision import transforms
from utils.utils_data import distort_decam_images, get_distortion_names,get_decam_distortions_composition,li_distort,li_distort_names

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
                 max_distortions: int = 2,
                 num_levels: int = 5,
                 pristine_prob: float = 0.05):

        # Convert paths to Path objects
        ref_images_csv_path = "/global/homes/s/salmab/ARNIQA/data/ref_images.csv"
        ref_images = pd.read_csv(ref_images_csv_path)
        hdu_numbers_csv_path = "/global/homes/s/salmab/ARNIQA/data/hdu_numbers.csv"
        hdu_numbers = pd.read_csv(hdu_numbers_csv_path)
     
        ref_images = ref_images['ref_images'].tolist()
        hdu_numbers = hdu_numbers['hdu_numbers'].tolist()

        self.ref_images = [Path(path) for path in ref_images]
        self.hdu_numbers = [hdu for hdu in hdu_numbers]

        self.patch_size = patch_size
        self.max_distortions = max_distortions
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
        
        # Define the RandomCrop transformation
        random_crop = transforms.RandomCrop(1024)
        
        img_A_orig = torch.tensor(img_A).unsqueeze(0)

        # Apply RandomCrop to the original image
        img_A_orig = random_crop(img_A_orig)
        
        img_B_orig = torch.tensor(img_B).unsqueeze(0)

        # Apply RandomCrop to the original image
        img_B_orig = random_crop(img_B_orig)        
        
        distort_functions_A = []
        distort_values_A = []
        distort_functions_B = []
        distort_values_B = []

        # Distort images with (1 - self.pristine_prob) probability for image A
        if random.random() > self.pristine_prob and self.max_distortions > 0:
            img_A_orig, distort_functions_A, distort_values_A = distort_decam_images(img_A_orig,
                                                                                 max_distortions=self.max_distortions,
                                                                                 num_levels=self.num_levels)

        # Use the same distortions for image B
        img_B_orig, distort_functions_B, distort_values_B = distort_decam_images(img_B_orig, distort_functions=distort_functions_A, distort_values=distort_values_A)


        distort_functions_A_names = get_distortion_names(distort_functions_A, li_distort_names)
        distort_functions_B_names = get_distortion_names(distort_functions_B, li_distort_names)

        # Pad to make the length of distort_functions and distort_values equal for all samples
        distort_functions_A_names += [""] * (self.max_distortions - len(distort_functions_A_names))
        distort_values_A += [torch.inf] * (self.max_distortions - len(distort_values_A))

        distort_functions_B_names += [""] * (self.max_distortions - len(distort_functions_B_names))
        distort_values_B += [torch.inf] * (self.max_distortions - len(distort_values_B))

       

        return {
            "img_A_orig": img_A_orig,"img_B_orig": img_B_orig,
            "distortion_functions_A": distort_functions_A, "distortion_values_A": distort_values_A,
            "distortion_functions_B": distort_functions_B, "distortion_values_B": distort_values_B
        }

    
    def __len__(self) -> int:
        return len(self.ref_images)


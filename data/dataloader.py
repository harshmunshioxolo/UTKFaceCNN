import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

class UTKDataset(Dataset):
    def __init__(self, root_dir:str) -> None:
        """DataLoader for the UTK Face dataset

        Summary:
            The dataset is given on kaggle, link: https://www.kaggle.com/jangedoo/utkface-new
            The dataset is in the form of cropped (and aligned) images. In case one wants to use the base image
                it can be used too. The facial landmarks are also given for the face alignment.
            
            NOTE: This dataset assumes input images as aligned.
                  The image files are named [age]_[gender]_[race]_[date&time].jpg
                  * [age] is an integer from 0-116
                  * [gender] is a boolean, 0: male, 1: Female
                  * [race] is an integer between 0-4, 0: white, 1: black, 2: Asian, 3: Indian, 4: Others
                  * [date&time] when the photo was clicked

        Args:
            root_dir (str): root directory
        """
        super(UTKDataset, self).__init__()
        if not isinstance(root_dir, str):
            raise TypeError("Expected root_dir to be of type str!")

        self.root_dir = root_dir
        self.cropped_data_dir = str(Path(root_dir) / "utkface_aligned_cropped" / "UTKFace")
        self.images = []
        self.age = []
        self.gender = []
        self.race = []

        image_list = [str(c) for c in Path(self.cropped_data_dir).iterdir()]
        for image in tqdm(image_list):
            # 0. Read and append the images
            self.images.append(np.array(Image.open(image)))
            
            # 1. extract the age, gender and race of that particular image
            age, gender, race = image.split("/")[-1].split("_")[0:3]
            self.age.append(age)
            self.gender.append(gender)
            self.race.append(race)
        
        # mandatory checks
        assert(len(self.images)==len(self.age))
        assert(len(self.images)==len(self.gender))
        assert(len(self.images)==len(self.race))
        print(">>> Assertion Passed <<<")
    
    def __len__(self) -> int:
        return len(self.images)

if __name__=="__main__":
    utkdataset = UTKDataset(root_dir="/scratch/hmunsh2s")
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
    def __init__(self, root_dir:str, transform=None) -> None:
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
        self.transform = transform

        image_list = [str(c) for c in Path(self.cropped_data_dir).iterdir()]
        for image in tqdm(image_list):
            # 0. Read and append the images
            self.images.append(np.array(Image.open(image)))
        
            # 1. extract the age, gender and race of that particular image
            image_metadata = image.split("/")[-1].split("_")
            try:
                age = int(image_metadata[0])
                gender = int(image_metadata[1])
                race = int(image_metadata[2])

                self.age.append(age)
                self.gender.append(gender)
                self.race.append(race)
            except:
                self.images.pop()

        # mandatory checks
        assert(len(self.images)==len(self.age))
        assert(len(self.images)==len(self.gender))
        assert(len(self.images)==len(self.race))
        print(">>> Assertion Passed <<<")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index) -> dict:
        #0. Fetch the image from the list and normalize it
        image = self.images[index]
        image = image / 255.0

        #1. Transpose the image into (R,G,B) format and convert it to tensor
        # image = image.transpose((2, 0, 1))
        image = transforms.ToTensor()(image)

        #2. Read the age gender race at the index
        gender = self.gender[index]
        # age = torch.Tensor(self.age[index])
        age = self.age[index]
        race = self.race[index]
        

        return {"image": image, "age": age,  "gender": gender, "race": race}

if __name__=="__main__":
    utkdataset = UTKDataset(root_dir="/scratch/hmunsh2s")
    utkloader = DataLoader(utkdataset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(utkloader):
        print(data['image'].shape)
        print(data['age'])
        print(data['gender'])
        print(data['race'])
        break
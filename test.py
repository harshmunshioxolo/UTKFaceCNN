import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from data.dataloader import UTKDataset
from PIL import Image
import torch.nn.functional as F
import numpy as np

race_classes = ["white", "black", "asian", "indian", "other"]
gender_classes = ["male", "female"]

utkdataset = UTKDataset(root_dir="/data/harsh/Documents/Experiments/pytorch_cpp/face_data")

if __name__=="__main__":
    # 0. Define the network architecture
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    print(">>> Initialized Resnet <<<")

    # Load the model
    net.load_state_dict(torch.load('UTKFaceCNN.pth'))
    print(">>> Loaded UTKFaceCNN Model <<<")

    # load the image
    image = np.array(Image.open("test_images/sample1.jpg"))
    image_norm = image / 255.0
    image_norm = torch.from_numpy(image_norm).view(1, 3, image_norm.shape[0], image_norm.shape[1]).float()
    # Forward inference
    output = net(image_norm)

    # put the network in evaluation mode
    net.eval()
    # Interpret the output
    # The first 5 outputs are race, next 1 is gender and last one in age
    sig = nn.Sigmoid()
    race_prob, race_idx = torch.max(F.softmax(output[:,0:5], 1), 1)
    gender_prob, gender_idx = torch.max(sig(output[:,5:6]), 1)
    age_op = output[:,6].squeeze(0).item()
    age = utkdataset.unnormalize(float(age_op))

    print(race_classes[race_idx], gender_classes[gender_idx], age)
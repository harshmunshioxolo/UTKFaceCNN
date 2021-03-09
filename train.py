from data.dataloader import UTKDataset
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

if __name__=="__main__":

    # -1: Define if device is cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0. define the network.
    net = models.resnet18(pretrained=True)

    # 1. change the final layer
    net.fc = nn.Linear(in_features=512, out_features=7, bias=True)
    net.to(device=device)

    # 2. Define the dataloader
    utkdataset = UTKDataset(root_dir="/home/harsh/Documents/github_projects/face_data")
    UTKLoader = DataLoader(utkdataset, batch_size=16, shuffle=True, num_workers=4)

    # 3. Identify the loss functions
    # The output layer consists of 8 outputs. 5 for race, 2 gender, 1 age = 8
    # For race => multiclass classification nn.CrossEntropyLoss()
    # For gender => binary classification BCEWithLogits()
    # For age => Standard regression loss.

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.BCELoss()
    criterion_3 = nn.L1Loss()

    # 4. Define the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 5. Activate sigmoid
    sig = nn.Sigmoid()

    # 5. Setup the net to train
    net.train()
    for epoch in tqdm(range(2)):
        for i, data in tqdm(enumerate(UTKLoader)):
            inputs = data["image"].to(device=device)
            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            race_label = data["race"].to(device=device)

            # forward to the net
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_1 = criterion_1(outputs[:, 0:5], race_label)
            loss_2 = criterion_2(sig(outputs[:, 5:6]), gender_label.unsqueeze(1).float())
            loss_3 = criterion_3(outputs[:, 6], age_label.float())
            loss = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()
    
    PATH = "./UTKFaceCNN.pth"
    torch.save(net.state_dict(), PATH)
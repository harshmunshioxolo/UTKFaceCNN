import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms, utils
from tqdm import tqdm

from data.dataloader import UTKDataset

if __name__ == "__main__":

    # -2: initiaze the summarywriter
    writer = SummaryWriter()

    # -1: Define if device is cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. define the network.
    net = models.resnet50(pretrained=True)

    # 1. change the final layer
    net.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    net.to(device=device)

    # 2a. Define the dataloader
    utkdataset = UTKDataset(
        root_dir="/data/harsh/Documents/Experiments/pytorch_cpp/face_data"
    )
    val_data = int(len(utkdataset) * 0.2)
    train_data_len = len(utkdataset) - val_data

    # 2b. Split the data in train and test
    train_data, test_data = random_split(utkdataset, [train_data_len, val_data])
    train_loader, val_loader = (
        DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4),
        DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4),
    )

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
    writer.add_graph(net, torch.randn(1, 3, 224, 224).to(device=device))

    # 5. Setup the net to train
    net.train()
    for epoch in tqdm(range(10)):
        for i, data in tqdm(enumerate(train_loader)):
            inputs = data["image"].to(device=device)
            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            race_label = data["race"].to(device=device)

            # forward to the net
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_1 = criterion_1(outputs[:, 0:5], race_label)
            loss_2 = criterion_2(
                sig(outputs[:, 5:6]), gender_label.unsqueeze(1).float()
            )
            loss_3 = criterion_3(outputs[:, 6], age_label.float())
            loss = loss_1 + loss_2 + loss_3
            writer.add_scalar(
                "Loss/Ethnicity", loss_1.item(), epoch * train_data_len + i
            )
            writer.add_scalar("Loss/Gender", loss_2.item(), epoch * train_data_len + i)
            writer.add_scalar("Loss/Age", loss_3.item(), epoch * train_data_len + i)
            writer.add_scalar(
                "Loss/total_loss", loss.item(), epoch * train_data_len + i
            )
            loss.backward()
            optimizer.step()

        # enter validation at evey 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(net.state_dict(), "./checkpoints/UTKFaceCNN_" + str(epoch+1) +".pth")
            print(">>>Validating<<<")
            # save the model
            for j, data in tqdm(enumerate(val_loader)):
                inputs = data["image"].to(device=device)
                age_label = data["age"].to(device=device)
                gender_label = data["gender"].to(device=device)
                race_label = data["race"].to(device=device)

                output = net(inputs)

                loss_1 = criterion_1(outputs[:, 0:5], race_label)
                loss_2 = criterion_2(
                    sig(outputs[:, 5:6]), gender_label.unsqueeze(1).float()
                )
                loss_3 = criterion_3(outputs[:, 6], age_label.float())
                loss = loss_1 + loss_2 + loss_3
                writer.add_scalar("Val/Ethnicity", loss_1.item(), j)
                writer.add_scalar("Val/Gender", loss_2.item(), j)
                writer.add_scalar("Val/Age", loss_3.item(), j)
                writer.add_scalar("Val/total_loss", loss.item(), j)

    PATH = "./UTKFaceCNN.pth"
    torch.save(net.state_dict(), PATH)

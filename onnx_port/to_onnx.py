import torch 
import torch.onnx
import torchvision.models as models
import torch.nn as nn

if __name__=="__main__":

    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
    print(">>> Initialized Resnet <<<")

    # Load the model
    net.load_state_dict(torch.load("../UTKFaceCNN.pth"))
    print(">>> Loaded UTKFaceCNN Model <<<")

    net.eval()

    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(net, x, "UTKFaceCNN.onnx", export_params=True)
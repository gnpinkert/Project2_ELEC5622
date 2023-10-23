import torch.nn as nn
import torch
import torch.hub
import torchvision.transforms
import ssl

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.fc1 = nn.Linear(1000, 6)

    def forward(self, x):
        x = self.alexnet(x)
        x = self.fc1(x)
        return x

def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    testmodel = AlexNet()

if __name__ == "__main__":
    main()
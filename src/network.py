import torch.nn as nn
import torch
import torch.hub
import torchvision.transforms
import ssl

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        self.alexnet.classifier[6] = nn.Linear(4096, 6, bias = True)

    def forward(self, x):
        x = self.alexnet(x)
        return x

def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    testmodel = AlexNet()
    pass

if __name__ == "__main__":
    main()
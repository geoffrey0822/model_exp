import torch, torchvision
import torch.nn as nn
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor()
])


class SiameseModel(nn.Module):
    def __init__(self, backbond, backbond_output, feature_output):
        super(SiameseModel, self).__init__()
        self.backbond = backbond
        self.backbond_output = backbond_output

        self.fc = nn.Sequential(nn.Linear(self.backbond_output,feature_output), nn.Sigmoid())

    def forward(self, x1):
        output = self.backbond(x1)
        output = output.view(output.size()[0], -1)

        output = self.fc(output)

        return output
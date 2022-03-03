import torch
import torch.nn as nn
import timm
import torchvision.models as models
import torch.nn.functional as F

def create_model(arch, pretrained, n_class):
    if arch == 'xception':
        return Xception(pretrained=pretrained, n_class=n_class)
    elif arch == 'googlenet':
        return GoogleNet(pretrained=pretrained, n_class=n_class)


class Xception(nn.Module):
    def __init__(self, arch = 'xception', pretrained = False, n_class = 1000):
        super(Xception, self).__init__()
        self.n_class = n_class
        model = timm.create_model('xception', pretrained=pretrained)

        # state_dict = torch.load('checkpoint/1000_class/model_best_all.pth')['state_dict']
        # # state_dict = torch.load('model_best_vehicles.pth')['state_dict']
        # state_dict = {k[7:]: v for k, v in state_dict.items()}
        # model.load_state_dict(state_dict)
        
        self.feature = torch.nn.Sequential(*(list(model.children())[:-1]))

        # for param in self.feature.parameters():
        #     param.requires_grad = False

        self.fc = nn.Linear(2048, n_class)

    def forward(self, x):
        return self.fc(self.feature(x))


class GoogleNet(nn.Module):
    def __init__(self, arch='googlenet', pretrained=False, n_class=1000):
        super(GoogleNet, self).__init__()
        self.n_class = n_class
        googlenet = models.googlenet(pretrained=pretrained)

        # state_dict = torch.load('checkpoint/1000_class/model_best_all.pth')['state_dict']
        # # state_dict = torch.load('model_best_vehicles.pth')['state_dict']
        # state_dict = {k[7:]: v for k, v in state_dict.items()}
        # model.load_state_dict(state_dict)

        self.feature = torch.nn.Sequential(*(list(googlenet.children())[:-1]))

        for param in self.feature.parameters():
            param.requires_grad = False


        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)

        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(2048, 1024)
        self.lin2 = torch.nn.Linear(1024, 512)
        self.lin3 = torch.nn.Linear(512, 256)
        self.lin4 = torch.nn.Linear(256, 64)
        self.lin5 = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.lin5(x)
        return x.squeeze(2)
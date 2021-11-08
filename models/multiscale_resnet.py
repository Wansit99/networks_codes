import torch
from torch import nn
from torchvision import models
from utils.utils import print_model

class multiscale_resnet(nn.Module):
    def __init__(self,num_class):
        super(multiscale_resnet,self).__init__()
        resnet50 =models.resnet50(pretrained=True)
        self.base_model =nn.Sequential(*list(resnet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(resnet50.fc.in_features,num_class)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp = nn.UpsamplingBilinear2d(size = (int(input_size*0.75)+1,  int(input_size*0.75)+1))

        x2 = self.interp(x)
        x = self.base_model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x2 = self.base_model(x2)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        out =[]
        out.append(self.classifier(x))
        out.append(self.classifier(x2))
        return out


if __name__=='__main__':
    model = multiscale_resnet(num_class=4)
    # print(model)
    #
    print_model(model)
    # input = torch.randn(1, 3, 224, 224)
    # out = model(input)
    # print(out.shape)
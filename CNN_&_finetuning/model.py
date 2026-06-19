import torch
import torch.nn as nn
import torchvision.models as models
import torchvision


class MyCNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结原模型参数，进行迁移学习
        for param in self.resnet18.parameters():
            param.requires_grad = False
        # 获取原有最后一层全连接层的输入特征数
        num_ftrs = self.resnet18.fc.in_features
        # 替换为适配十分类的全新linear层
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

        # 解冻最后两层参数，进行训练微调
        for param in self.resnet18.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet18(x)
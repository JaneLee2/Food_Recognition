import torch
from torch import nn
from utils.utils import Main


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        # 加载预训练的DenseNet121模型
        self.model = torch.torchvision.models.densenet121(pretrained=True)
        #
        # 读取输入特征的维度（因为这一维度不需要修改）
        num_ftrs = self.model.classifier.in_features
        # 修改最后一层的输出维度，即分类数
        self.model.classifier = nn.Linear(num_ftrs, num_classes)  # 根据你的任务设定正确的类别数

    def forward(self, x):
        x = self.model(x)
        return x


import torch
from torch import nn
import torchvision
from utils.utils import Main
from torchvision import transforms


class ResNet50(nn.Module):
    def __init__(self, num_cls=10, dropout_rate=0.5):
        super(ResNet50, self).__init__()

        # 加载预训练的resnet模型
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        # https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#ResNet50_Weights
        model_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # 分类数
        self.num_cls = num_cls
        # 获取resnet的卷积层和全连接层信息
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.fc = model_resnet.fc
        self.transforms = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        # 在全连接层之前添加Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 读取输入特征的维度（因为这一维度不需要修改）
        num_fc = self.fc.in_features
        # 修改最后一层的输出维度，即分类数
        self.fc = torch.nn.Linear(num_fc, self.num_cls)
        # 对于模型的每个权重，使其不进行反向传播，即固定参数
        for param in model_resnet.parameters():
            param.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.transforms(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    resnet_cnn = ResNet50(10)
    main_train = Main(device='cuda:0',
                      num_classes=10,
                      train_path=r"D:\DataSets\Chinese_Food\aglin_Chinesefood-10\train",
                      valid_path=r"D:\DataSets\Chinese_Food\aglin_Chinesefood-10\val",
                      batch_size=128,
                      lr=0.01,
                      epochs=90,
                      weight_path=r"../weights/best_model_resnet50_no_aug.pth",
                      log_file_path=r'../logs/Train_data_resnet50_no_aug.xlsx',
                      tensorbord_path=r'../logs/runs/resnet50_no_aug')
    main_train.train(resnet_cnn)

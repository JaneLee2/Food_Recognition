import torch
from torch import nn
import torchvision
from torchvision import transforms
from utils.utils import Main


class VGG19_BN(nn.Module):
    def __init__(self, num_cls=10, dropout_rate=0.5):
        super(VGG19_BN, self).__init__()

        # 加载预训练的VGG19-BN模型
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19_bn.html
        # https://pytorch.org/vision/0.12/_modules/torchvision/models/vgg.html
        model_vgg19_bn = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
        # 分类数
        self.num_cls = num_cls
        self.transforms = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        # 获取VGG19-BN的卷积层和全连接层
        self.features = model_vgg19_bn.features
        self.avgpool = model_vgg19_bn.avgpool
        self.classifier = model_vgg19_bn.classifier

        # 在全连接层之前添加Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 读取输入特征的维度（因为这一维度不需要修改）
        num_fc = self.classifier[6].in_features
        # 修改最后一层的输出维度，即分类数
        self.classifier[6] = torch.nn.Linear(num_fc, self.num_cls)
        # 对于模型的每个权重，使其不进行反向传播，即固定参数
        for param in model_vgg19_bn.parameters():
            param.requires_grad = False
        # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
        for param in self.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.transforms(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    vgg_cnn = VGG19_BN(208)
    main_train = Main(device='cuda:0',
                      num_classes=208,
                      train_path=r"D:\DataSets\Chinese_Food\aglin_release_data\train",
                      valid_path=r"D:\DataSets\Chinese_Food\aglin_release_data\val",
                      batch_size=126,
                      lr=0.01,
                      epochs=90,
                      weight_path=r"../weights/best_model_208_vgg191.pth",
                      log_file_path=r'../logs/Train_data_208_vgg191.xlsx',
                      tensorbord_path=r'../logs/runs/208_vgg191')
    main_train.train(vgg_cnn)

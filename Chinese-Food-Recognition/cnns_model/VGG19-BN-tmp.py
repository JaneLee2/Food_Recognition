import torch
from torch import nn
from dataset.food_dataset import Food_Loader
from models_tmp.config import Common, Train, Prints, Valid
from d2l import torch as d2l
import os
import torchvision
from torchvision import models



class VGG19_BN(nn.Module):
    def __init__(self, dropout_rate=0.5, num_cls=10):
        super(VGG19_BN, self).__init__()

        # 加载预训练的VGG19-BN模型
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19_bn.html
        model_vgg19_bn = torchvision.models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        # 分类数
        self.num_cls = num_cls
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, valid_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss = nn.CrossEntropyLoss()
    acc_high = 0
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        # 使用验证集挑选模型
        valid_acc = evaluate_accuracy_gpu(net, valid_iter)
        if valid_acc > acc_high:
            acc_high = valid_acc
            print('the highest acc value is:', str(acc_high))
            torch.save(net.state_dict(), os.path.join(Valid.valid_logDir, "simple_cnn_best_Epoch" + '.pth'))

        # 使用测试集测试泛化能力
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'train acc {train_acc:.3f}, valid acc {valid_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        with open(Valid.valid_logDir + '/valid.txt', 'a+') as f:
            f.write('epoch' + str(epoch) + ':' + 'the train acc is' + str(train_acc) + 'the valid acc is' + str(
                valid_acc) + 'the test acc is' + str(test_acc) + '\n')
        f.close()


if __name__ == "__main__":
    train_iter, valid_iter, test_iter = Food_Loader(datasets_path=Common.basePath,
                                                    batch_size=Train.batch_size,
                                                    num_workers = Train.num_workers,
                                                    train="train",
                                                    valid="val",
                                                    test="test").load_dataset()

    net = VGG19_BN(num_cls=10)
    train(net, train_iter, valid_iter, test_iter, Train.epochs, Train.lr, d2l.try_gpu())

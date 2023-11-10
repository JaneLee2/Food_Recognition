import torch
from torch import nn
from utils.utils import Main


class SimpleCnnNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCnnNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes))

        self.net = nn.Sequential(self.conv,
                                 self.fc)

        self.net_print()

    def forward(self, img):
        output = self.net(img)
        return output

    def net_print(self):
        print("----------Net Info------------------- ")
        X = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print("----------CONV Net Info------------------- ")
        X = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
        for layer in self.conv:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)


if __name__ == '__main__':
    simple_cnn = SimpleCnnNet(10)
    main_train = Main(device='cuda:0',
                      num_classes=10,
                      train_path=r"D:\DataSets\Chinese_Food\Chinesefood-10\train",
                      valid_path=r"D:\DataSets\Chinese_Food\Chinesefood-10\val",
                      batch_size=128,
                      lr=0.01,
                      epochs=90,
                      weight_path=r"../weights/best_model_simple_cnn1.pth",
                      log_file_path=r'../logs/Train_data_simple_cnn1.xlsx',
                      tensorbord_path=r'../logs/runs/simple_cnn1')
    main_train.train(simple_cnn)


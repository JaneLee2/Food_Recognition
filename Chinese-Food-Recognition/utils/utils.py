import sys
from tqdm import tqdm
import xlwt
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset.food_dataset import ImageFolder


class Main:
    def __init__(self, device='cuda:0',
                 num_classes=10,
                 train_path=r"D:\DataSets\Chinese_Food\Chinesefood-10\train",
                 valid_path=r"D:\DataSets\Chinese_Food\Chinesefood-10\val",
                 batch_size=128,
                 lr=0.01,
                 epochs=90,
                 weight_path=r"../weights/best_model_vgg19.pth",
                 log_file_path=r'../logs/vgg/Train_data_vgg19.xlsx',
                 tensorbord_path=r'../logs/runs/vgg19'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_path = weight_path
        self.log_file_path = log_file_path
        self.tensorbord_path = tensorbord_path
        self.tb_writer = SummaryWriter(self.tensorbord_path)
        self.init_fold()
        self.init_xlsx()

        self.data_transform = {
            "train": transforms.Compose([transforms.ToTensor()]),
            "val": transforms.Compose([transforms.ToTensor()])}
        self.load_food_dataset()

    def init_fold(self):
        if not os.path.exists(os.path.split(self.weight_path)[0]):
            os.makedirs(os.path.split(self.weight_path)[0])
        if not os.path.exists(os.path.split(self.log_file_path)[0]):
            os.makedirs(os.path.split(self.log_file_path)[0])
        if not os.path.exists(os.path.split(self.tensorbord_path)[0]):
            os.makedirs(os.path.split(self.tensorbord_path)[0])

    def init_xlsx(self):
        self.book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
        # 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
        self.sheet1 = self.book.add_sheet(u'Train_data', cell_overwrite_ok=True)
        # 向表中添加数据
        self.sheet1.write(0, 0, 'epoch')
        self.sheet1.write(0, 1, 'Train_Loss')
        self.sheet1.write(0, 2, 'Train_Acc')
        self.sheet1.write(0, 3, 'Val_Loss')
        self.sheet1.write(0, 4, 'Val_Acc')
        self.sheet1.write(0, 5, 'lr')
        self.sheet1.write(0, 6, 'Best val Acc')

    def load_food_dataset(self):
        train_dataset = ImageFolder(self.train_path, transform=self.data_transform["train"])
        val_dataset = ImageFolder(self.valid_path, transform=self.data_transform["val"])
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=0)

    def train_one_epoch(self, model, optimizer, data_loader, epoch):
        model.train()
        loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        accu_loss = torch.zeros(1).to(self.device)  # 累计损失
        accu_num = torch.zeros(1).to(self.device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            sample_num += images.shape[0]

            pred = model(images)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()

            loss = loss_function(pred, labels)
            loss.backward()
            accu_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num

    @torch.no_grad()
    def evaluate(self, model, data_loader, epoch):
        loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        model.eval()

        accu_num = torch.zeros(1).to(self.device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(self.device)  # 累计损失

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            sample_num += images.shape[0]

            pred = model(images)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()

            loss = loss_function(pred, labels)
            accu_loss += loss

            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num)

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num

    def train(self, net):
        best_acc = 0
        model = net.to(self.device)
        images = torch.zeros(1, 3, 224, 224).to(self.device)  # 要求大小与输入图片的大小一致
        self.tb_writer.add_graph(model, images, verbose=False)
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=self.lr, momentum=0.9, weight_decay=5E-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        for epoch in range(self.epochs):
            # train
            train_loss, train_acc = self.train_one_epoch(model, optimizer, self.train_loader, epoch)
            scheduler.step()
            # validate
            val_loss, val_acc = self.evaluate(model, self.val_loader, epoch)
            # 记录训练过程数据到文件
            self.sheet1.write(epoch + 1, 0, epoch + 1)
            self.sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))
            self.sheet1.write(epoch + 1, 1, str(round(train_loss, 3)))
            self.sheet1.write(epoch + 1, 2, str(round(train_acc, 3)))
            self.sheet1.write(epoch + 1, 3, str(round(val_loss, 3)))
            self.sheet1.write(epoch + 1, 4, str(round(val_acc, 3)))
            # 记录训练过程数据到Tensorboard
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            self.tb_writer.add_scalar(tags[0], train_loss, epoch)
            self.tb_writer.add_scalar(tags[1], train_acc, epoch)
            self.tb_writer.add_scalar(tags[2], val_loss, epoch)
            self.tb_writer.add_scalar(tags[3], val_acc, epoch)
            self.tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), self.weight_path)

        self.sheet1.write(1, 6, str(best_acc))
        self.book.save(self.log_file_path)
        print("The Best Acc = : {:.4f}".format(best_acc))

import os

from model.simple_cnn_tmp import SimpleCnnNet
import torch
from torchvision import transforms
from torch.autograd import Variable
import cv2


class FoodRecognition:
    def __init__(self, food_recognition_model=r"../weights/best_model_simple_cnn.pth"):
        self.food_recognition = SimpleCnnNet(10)
        if torch.cuda.is_available():
            self.food_recognition.cuda()
            checkpoint = torch.load(food_recognition_model)
        else:
            checkpoint = torch.load(food_recognition_model, torch.device('cpu'))
        self.food_recognition.load_state_dict(checkpoint)
        self.food_recognition.eval()

        self.test_augs = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def food_recognition_inference(self, image):
        img = self.test_augs(image)
        # 3维扩4维度
        inputs_test = img.unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        prediction = self.food_recognition(inputs_test)
        pred_classes = torch.max(prediction, dim=1)[1]
        print("prediction:", prediction)
        print("pred_classes:", pred_classes)
        return pred_classes


if __name__ == '__main__':
    folder_path = r"D:\DataSets\Chinese_Food\Chinesefood-10\tests\000"
    to_path = r"../results/tests/000"
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    food_engine = FoodRecognition()
    for image_path in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, image_path))
        pred_classes = food_engine.food_recognition_inference(img)
        cv2.putText(img, pred_classes, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        cv2.imwrite(os.path.join(to_path, image_path), img)

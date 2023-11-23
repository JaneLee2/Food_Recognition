import os
from torchvision import transforms
from PIL import Image

food_dataset_path = r"D:\DataSets\Chinese_Food\release_data"
align_dataset_path = r"D:\DataSets\Chinese_Food\aglin_release_data"

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224)])
# for folder in ["train", "tests", "val"]:
for folder in ["tests", "val"]:
    child_food_folder = os.path.join(food_dataset_path, folder)
    child_align_folder = os.path.join(align_dataset_path, folder)
    for class_id in os.listdir(child_food_folder):
        class_food_folder = os.path.join(child_food_folder, class_id)
        class_align_folder = os.path.join(child_align_folder, class_id)
        if not os.path.exists(class_align_folder):
            os.makedirs(class_align_folder)
        for img_id in os.listdir(class_food_folder):
            img_food_path = os.path.join(class_food_folder, img_id)
            img_align_path = os.path.join(class_align_folder, os.path.splitext(img_id)[0] + ".png")
            print(img_food_path)
            image = Image.open(img_food_path)
            align_image = transform(image)
            if align_image.mode == "CMYK":
                align_image = align_image.convert('RGB')
            align_image.save(img_align_path)

import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import cv2
from torchvision import transforms
from utils import (
    AWA2_PATH,
    PREDICATE_BINARY_MAT_PATH,
    ALL_CLASS_PATH,
    JPEG_PATH,
    mapping_class_to_index,
)


class AnimalDataset(data.dataset.Dataset):
    """
    Copied from: https://github.com/dfan/awa2-zero-shot-learning/blob/master/AnimalDataset.py
    """

    def __init__(self, classes_path, transform=None):
        # root directory path for awa2 dataset
        self.awa2_path = AWA2_PATH
        self.predicate_binary_mat_path = PREDICATE_BINARY_MAT_PATH
        self.all_classes_path = ALL_CLASS_PATH
        self.JPEG_path = JPEG_PATH

        # one-hot encoded class labels by attributes
        predicate_binary_mat = np.array(
            np.genfromtxt(self.awa2_path + "predicate-matrix-binary.txt", dtype="int")
        )
        self.predicate_binary_mat = predicate_binary_mat
        # image transformer
        self.transform = transform

        self.class_to_index = mapping_class_to_index()

        img_names = []
        img_class = []
        with open(self.awa2_path + classes_path) as f:
            for line in f:
                class_name = line.strip()
                FOLDER_DIR = os.path.join(self.awa2_path + self.JPEG_path, class_name)
                file_descriptor = os.path.join(FOLDER_DIR, "*.jpg")
                files = glob(file_descriptor)

                class_index = self.class_to_index[class_name]
                for file_name in files:
                    img_names.append(file_name)
                    img_class.append(class_index)
        self.img_names = img_names
        self.img_class = img_class

    def __getitem__(self, index):
        im = Image.open(self.img_names[index])
        if im.getbands()[0] == "L":
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        if im.shape != (3, 224, 224):
            print(self.img_names[index])

        im_class = self.img_class[index]
        im_predicate = self.predicate_binary_mat[im_class, :]
        return im, im_predicate, self.img_names[index], im_class

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    train_process_steps = transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Resize((224, 224)),  # ImageNet standard
            transforms.ToTensor(),
        ]
    )
    awa2 = AnimalDataset("trainclasses.txt", train_process_steps)
    awa2[0]

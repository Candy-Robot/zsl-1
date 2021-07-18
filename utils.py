import torchvision
import torch.nn as nn
from torchvision import transforms
import torch
import sys
import numpy as np


# path constants
AWA2_PATH = "/home/fangyuan/Animals_with_Attributes2/"
PREDICATE_BINARY_MAT_PATH = "predicate-matrix-binary.txt"
ALL_CLASS_PATH = "classes.txt"
TRAIN_CLASS_PATH = "trainclasses.txt"
TEST_CLASS_PATH = "testclasses.txt"
JPEG_PATH = "JPEGImages"


# training image transformer
train_transformer = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
    ]
)


# testing image transformer
test_transformer = transforms.Compose(
    [transforms.Resize((244, 244)), transforms.ToTensor()]
)


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype="uint8")[y]


def CUDA(tensor: torch.Tensor):
    """Move tensor to cuda if cuda is available."""
    return tensor.cuda() if torch.cuda.is_available() else tensor


def get_res50_model():
    """Get a pre-trained resnet49 model."""
    res50 = torchvision.models.resnet50(pretrained=True)
    # remove final classifier
    # res50 = nn.Sequential(*list(res50.children())[:-1])
    # don't update params in pre-trained resnet50
    for _, param in res50.named_parameters():
        param.requires_grad = False
    return res50


def build_model(num_outputs):
    """Build model which takes image as input, and label embedding as ouput."""
    model = get_res50_model()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(num_features, num_outputs),
    )
    return model


def mapping_class_to_index():
    """
    Build dictionary of indices to classes
    """
    class_to_index = dict()
    with open(AWA2_PATH + ALL_CLASS_PATH) as f:
        index = 0
        for line in f:
            class_name = line.split("\t")[1].strip()
            class_to_index[class_name] = index
            index += 1
    return class_to_index


def mapping_index_to_class():
    """
    Build dictionary of indices to classes
    """
    index_to_class = []
    with open(AWA2_PATH + ALL_CLASS_PATH) as f:
        for line in f:
            class_name = line.split("\t")[1].strip()
            index_to_class.append(class_name)
    return index_to_class


def load_model(model_path, num_labels):
    """Load model from model_path."""
    model = CUDA(build_model(num_labels))
    model.load_state_dict(torch.load(model_path))
    return model


def get_predicate_binary_mat():
    """Get label predicate binary matrix."""
    return np.array(np.genfromtxt(AWA2_PATH + PREDICATE_BINARY_MAT_PATH, dtype="int"))


def get_all_classes():
    """Get all classes."""
    return np.array(np.genfromtxt(AWA2_PATH + ALL_CLASS_PATH, dtype="str"))[:, -1]


def get_train_classes():
    """Get training classes."""
    return np.array(np.genfromtxt(AWA2_PATH + TRAIN_CLASS_PATH, dtype="str"))


def get_test_classes():
    """Get test classes."""
    return np.array(np.genfromtxt(AWA2_PATH + TEST_CLASS_PATH, dtype="str"))


def get_hamming_dist(curr_labels, class_labels):
    return np.sum(curr_labels != class_labels)


def get_cosine_dist(curr_labels, class_labels):
    return (
        np.sum(curr_labels * class_labels)
        / np.sqrt(np.sum(curr_labels))
        / np.sqrt(np.sum(class_labels))
    )


def get_euclidean_dist(curr_labels, class_labels):
    return np.sqrt(np.sum((curr_labels - class_labels) ** 2))


def find_best_pred_class(pred_labels, predicate_binary_mat, all_classes, train_classes):
    """Find best class for predicted labels by find minimal distance between predicted value
    and the true value."""
    predictions = []
    for i in range(pred_labels.shape[0]):
        curr_labels = pred_labels[i, :].cpu().detach().numpy()
        best_dist = sys.maxsize
        best_index = -1
        for j in range(predicate_binary_mat.shape[0]):
            class_labels = predicate_binary_mat[j, :]
            # get euclidean distance
            dist = get_euclidean_dist(curr_labels, class_labels)
            if dist < best_dist and all_classes[j] not in train_classes:
                best_index = j
                best_dist = dist
        predictions.append(all_classes[best_index])
    return predictions

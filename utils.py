import torchvision
import torch.nn as nn
from torchvision import transforms
import torch
import sys
import numpy as np

import matplotlib.pyplot as plt


# path constants
AWA2_PATH = "/home/tangyicheng/awa2/"
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


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("grad_flow.png")

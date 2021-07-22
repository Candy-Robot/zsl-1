from operator import index
from torchvision import transforms
from utils import *
from awa2_dataset import AnimalDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import sys
import torch
import numpy as np


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


def load_model(model_path, num_labels):
    """Load model from model_path."""
    model = CUDA(build_model(num_labels))
    model.load_state_dict(torch.load(model_path))
    return model


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


def train(num_outputs, epochs, eval_interval, lr, batch_size, criterion):
    """Train awa2 model.

    Args:
        num_outputs (int): model output dims
        epochs (int): train epochs
        eval_interval (int): validation interval
        lr (float): learning rate
        batch_size (int): batch size
        criterion (func): loss function
    """
    train_dataset = AnimalDataset(TRAIN_CLASS_PATH, transform=train_transformer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    total_steps = len(train_dataloader)
    model = CUDA(build_model(num_outputs))
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (imgs, img_predicates, img_names, img_classes) in enumerate(
            train_dataloader
        ):
            if imgs.shape[0] < 2:
                break
            imgs = CUDA(imgs)
            # take img_predicate as embedding of the labels
            # or we can also use glove embedding of labels
            img_predicates = CUDA(img_predicates.to(torch.float32))
            model.train()

            outputs = model(imgs)
            sigmoid_outputs = torch.sigmoid(outputs)
            loss = criterion(sigmoid_outputs, img_predicates)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                curr_iter = epoch * len(train_dataloader) + i
                print(
                    "Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_steps, loss.item()
                    )
                )
                sys.stdout.flush()

        # Do some evaluations
        if (epoch + 1) % eval_interval == 0:
            print("Evaluating:")
            curr_acc = evaluate(model)
            print(
                "Epoch [{}/{}] Approx. training accuracy: {}".format(
                    epoch + 1, epochs, curr_acc
                )
            )

    # save model
    torch.save(model.state_dict(), "models/{}".format("awa2-model.bin"))
    torch.save(optimizer.state_dict(), "models/{}".format("awa2-optim-model.bin"))


def evaluate(model):
    test_dataset = AnimalDataset(TEST_CLASS_PATH, transform=test_transformer)
    test_dataloader = DataLoader(test_dataset)
    model.eval()

    pred_classes = []
    truth_classes = []

    predicate_binary_mat = get_predicate_binary_mat()
    all_classes = get_all_classes()
    train_classes = get_train_classes()

    with torch.no_grad():
        for i, (imgs, img_predicates, img_names, img_classes) in enumerate(
            test_dataloader
        ):
            imgs = CUDA(imgs)
            img_predicates = CUDA(img_predicates)
            outputs = model(imgs)
            # make outputs limit in 0 to 1
            sigmoid_outputs = torch.sigmoid(outputs)
            pred_labels = sigmoid_outputs  # > 0.5
            # find bes predict class
            curr_pred_classes = find_best_pred_class(
                pred_labels, predicate_binary_mat, all_classes, train_classes
            )
            pred_classes.extend(curr_pred_classes)

            curr_truth_classes = []
            for index in img_classes:
                curr_truth_classes.append(all_classes[index])
            truth_classes.extend(curr_truth_classes)

    pred_classes = np.array(pred_classes)
    truth_classes = np.array(truth_classes)
    mean_acc = np.mean(pred_classes == truth_classes)

    # Reset
    model.train()
    return mean_acc


def test(model_path, num_labels, output_filename):
    test_dataset = AnimalDataset(TEST_CLASS_PATH, transform=test_transformer)
    test_dataloader = DataLoader(test_dataset)

    model = load_model(model_path, num_labels)
    model.eval()

    predicate_binary_mat = get_predicate_binary_mat()
    all_classes = get_all_classes()
    train_classes = get_train_classes()
    class_to_index = mapping_class_to_index()

    pred_classes = []
    output_img_names = []
    success_cases = 0
    total_cases = len(test_dataset)

    with torch.no_grad():
        for i, (imgs, img_predicates, img_names, img_classes) in enumerate(
            test_dataloader
        ):
            imgs, img_predicates = CUDA(imgs), CUDA(img_predicates).float()
            outputs = model(imgs)
            sigmoid_outputs = torch.sigmoid(outputs)
            pred_labels = sigmoid_outputs  # > 0.5
            curr_pred_classes = find_best_pred_class(
                pred_labels, predicate_binary_mat, all_classes, train_classes
            )
            pred_classes.extend(curr_pred_classes)
            output_img_names.extend(img_names)

            if class_to_index[curr_pred_classes[0]] == img_classes.item():
                success_cases += 1

            if i % 1000 == 0:
                print("Prediction iter: {}".format(i))

    with open(output_filename, "w") as f:
        for i in range(len(pred_classes)):
            output_name = output_img_names[i].replace(AWA2_PATH + JPEG_PATH, "")
            f.write(output_name + " " + pred_classes[i] + "\n")
        f.write(
            "Success Cases: {0}\tTotal Cases: {1}".format(success_cases, total_cases)
        )


def test_single_image(model_path, num_labels, image_index):
    predicate_binary_mat = get_predicate_binary_mat()
    all_classes = get_all_classes()
    train_classes = get_train_classes()

    test_dataset = AnimalDataset(TEST_CLASS_PATH, transform=test_transformer)
    img, img_predicate, img_name, img_class = test_dataset[image_index]

    model = load_model(model_path, num_labels)
    model.eval()

    input_img = img.unsqueeze(dim=0)
    outputs = model(CUDA(input_img))
    sigmoid_outputs = torch.sigmoid(outputs)

    pred_class = find_best_pred_class(
        sigmoid_outputs, predicate_binary_mat, all_classes, train_classes
    )

    index_to_class = mapping_index_to_class()

    result = (
        "Test Image Path: {0}\nTest Image Class: {1}\nPredict Image Class: {2}".format(
            img_name, index_to_class[img_class], pred_class[0]
        )
    )
    return img, result


if __name__ == "__main__":
    criterion = nn.BCELoss()
    # train(85, 25, 1, 0.000025, 24, criterion)
    test("models/awa2-model.bin", 85, "awa2-test.txt")
    # test_single_image("models/awa2-model.bin", 85, 22)

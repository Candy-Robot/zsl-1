import sys

from awa2_dataset import AnimalDataset
from awa2_graph_dataset import AwA2GraphDataset
from networkx.drawing.nx_pylab import draw
from utils import (
    TEST_CLASS_PATH,
    TRAIN_CLASS_PATH,
    AWA2_PATH,
    JPEG_PATH,
    get_euclidean_dist,
    get_predicate_binary_mat,
    get_res50_model,
    get_test_classes,
    mapping_class_to_index,
    mapping_index_to_class,
    to_categorical,
    train_transformer,
    test_transformer,
    CUDA,
    plot_grad_flow,
)
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter

import networkx as nx


def find_best_pred_class_index(label_embeddings, img_embedding, test_class_indexes):
    best_dist = sys.maxsize
    best_index = -1
    for i in range(len(label_embeddings)):
        if i not in test_class_indexes:
            continue
        curr_label_embedding = label_embeddings[i, :].cpu().detach().numpy()
        dist = get_euclidean_dist(
            curr_label_embedding, img_embedding.cpu().detach().numpy()
        )
        if dist < best_dist:
            best_index = i
            best_dist = dist
    return best_index


class AwA2Conv(nn.Module):
    def __init__(self, num_feature, num_output):
        super(AwA2Conv, self).__init__()
        self.conv1 = GraphConv(num_feature, 1024)
        self.conv2 = GraphConv(1024, 512)
        self.conv3 = GraphConv(512, 256)
        self.conv4 = GraphConv(256, 128)
        self.conv5 = GraphConv(128, num_output)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        h = F.leaky_relu(h)
        h = self.conv4(g, h)
        h = F.leaky_relu(h)
        h = self.conv5(g, h)
        g.ndata["h"] = h
        return h
        # return dgl.mean_nodes(g, "h")


def build_model(dim_resnet50_feat, dim_label):
    # resnet50 -> 2048 -> 2048x40 -> 40
    #                        |
    #                       gcn
    final_untrained_fc = nn.Linear(dim_resnet50_feat, dim_label, bias=False)
    res50 = get_res50_model()
    # model = nn.Sequential(*list(res50.children())[:-1])
    res50._modules["fc"] = final_untrained_fc
    return res50


def train(lr, batch_size, epochs):
    writer = SummaryWriter(log_dir="./log/zsl_graph/")
    dataset = AwA2GraphDataset()
    g = dataset[0]

    dim_label_feat = g.ndata["feat"].shape[1]
    dim_res50_feat = 2048
    dim_label = 50
    gcn_model = AwA2Conv(dim_label_feat, dim_res50_feat)
    # res_model = build_model(dim_res50_feat, dim_label)
    res50 = get_res50_model()
    res_model = nn.Sequential(*list(res50.children())[:-1])

    train_dataset = AnimalDataset(TRAIN_CLASS_PATH, transform=train_transformer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)

    total_steps = len(train_dataloader)

    gcn_model.train()

    for epoch in range(epochs):
        for i, (imgs, img_predicates, img_names, img_classes) in enumerate(
            train_dataloader
        ):
            if imgs.shape[0] < 2:
                break
            # take img_predicate as embedding of the labels
            # or we can also use glove embedding of labels

            gcn_outputs = gcn_model(g, g.ndata["feat"])
            # set resnet50 fc weights to gcn outputs
            # train_outputs = gcn_outputs[g.ndata["train_mask"]]
            # res_model._modules["fc"].weight.data = gcn_outputs

            # true_labels = F.one_hot(img_classes, num_classes=dim_label)
            img_features = res_model(imgs).squeeze()
            img_preds = torch.matmul(img_features, gcn_outputs.T)
            softmax_img_preds = F.softmax(img_preds, dim=1)
            optimizer.zero_grad()
            loss = F.cross_entropy(softmax_img_preds, img_classes)
            loss.backward()
            # plot_grad_flow(gcn_model.named_parameters())
            optimizer.step()

            # histogram graph for parameters of networks
            for name, param in gcn_model.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), total_steps
                )

            if i % 50 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_steps, loss.item()
                    )
                )
                writer.add_scalar("Loss", loss.item(), epoch * total_steps + i)
                sys.stdout.flush()

    # save model
    torch.save(gcn_model.state_dict(), "models/{}".format("awa2-gcn-model.bin"))


def test(output_filename):
    dataset = AwA2GraphDataset()
    g = dataset[0]

    dim_label_feat = g.ndata["feat"].shape[1]
    dim_res50_feat = 2048
    dim_label = 50
    gcn_model = AwA2Conv(dim_label_feat, dim_res50_feat)
    # load pretrained model
    gcn_model.load_state_dict(torch.load("models/awa2-gcn-model.bin"))

    res50 = get_res50_model()
    res_model = nn.Sequential(*list(res50.children())[:-1])

    test_dataset = AnimalDataset(TEST_CLASS_PATH, transform=test_transformer)
    test_dataloader = DataLoader(test_dataset)

    gcn_model.eval()

    gcn_outputs = gcn_model(g, g.ndata["feat"])

    success_cases = 0
    pred_img_names = []
    output_img_names = []
    total_cases = len(test_dataset)

    test_classes = get_test_classes()
    class_to_index = mapping_class_to_index()
    test_class_indexes = []
    for tc in test_classes:
        test_class_indexes.append(class_to_index[tc])

    for i, (imgs, img_predicates, img_names, img_classes) in enumerate(test_dataloader):
        output_img_names.extend(img_names)
        # take img_predicate as embedding of the labels
        # or we can also use glove embedding of labels
        # set resnet50 fc weights to gcn outputs
        # train_outputs = gcn_outputs[g.ndata["train_mask"]]
        # res_model._modules["fc"].weight.data = gcn_outputs

        # true_labels = F.one_hot(img_classes, num_classes=dim_label)
        imgs_feature = res_model(imgs).squeeze()
        pred_class = find_best_pred_class_index(
            gcn_outputs, imgs_feature, test_class_indexes
        )
        pred_img_names.append(mapping_index_to_class()[pred_class])

        if pred_class == img_classes.item():
            success_cases += 1

    with open(output_filename, "w") as f:
        for i in range(len(pred_img_names)):
            output_name = output_img_names[i].replace(AWA2_PATH + JPEG_PATH, "")
            f.write(output_name + " " + pred_img_names[i] + "\n")
        f.write(
            "Success Cases: {0}\tTotal Cases: {1}".format(success_cases, total_cases)
        )


if __name__ == "__main__":
    train(lr=0.0001, batch_size=24, epochs=25)
    # test(output_filename="zsl-graph-test.txt")

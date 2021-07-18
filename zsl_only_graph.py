import sys

from awa2_dataset import AnimalDataset
from awa2_graph_dataset import AwA2GraphDataset
from networkx.drawing.nx_pylab import draw
from utils import (
    TRAIN_CLASS_PATH,
    get_predicate_binary_mat,
    get_res50_model,
    to_categorical,
    train_transformer,
    CUDA,
)
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import networkx as nx


class AwA2Conv(nn.Module):
    def __init__(self, num_feature, num_output):
        super(AwA2Conv, self).__init__()
        self.conv1 = GraphConv(num_feature, 256)
        self.conv2 = GraphConv(256, num_output)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
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
    dataset = AwA2GraphDataset()
    g = dataset[0]

    dim_label_feat = g.ndata["feat"].shape[1]
    dim_res50_feat = 2048
    dim_label = 50
    dim_train_label = 40
    gcn_model = AwA2Conv(dim_label_feat, dim_res50_feat)
    res_model = build_model(dim_res50_feat, dim_label)

    train_dataset = AnimalDataset(TRAIN_CLASS_PATH, transform=train_transformer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)

    total_steps = len(train_dataloader)

    for epoch in range(epochs):
        for i, (imgs, img_predicates, img_names, img_classes) in enumerate(
            train_dataloader
        ):
            if imgs.shape[0] < 2:
                break
            # take img_predicate as embedding of the labels
            # or we can also use glove embedding of labels
            gcn_model.train()

            gcn_outputs = gcn_model(g, g.ndata["feat"])
            # set resnet50 fc weights to gcn outputs
            # train_outputs = gcn_outputs[g.ndata["train_mask"]]
            res_model._modules["fc"].weight.data = gcn_outputs

            true_labels = F.one_hot(img_classes, num_classes=dim_label)
            preds = res_model(imgs)
            loss = F.cross_entropy(preds, img_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_steps, loss.item()
                    )
                )
                sys.stdout.flush()

    # save model
    torch.save(gcn_model.state_dict(), "models/{}".format("awa2-gcn-model.bin"))


if __name__ == "__main__":
    train(lr=0.000025, batch_size=24, epochs=25)

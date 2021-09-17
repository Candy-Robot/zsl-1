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
    transfor_matrix_binary,
)
from utils import *

import numpy as np
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
from dgl.data import DGLDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import random
from build_graph import build_transform_edges


import networkx as nx


class AwA2GraphDataset(DGLDataset):
    def __init__(self):
        # self.glove = GloVe("materials/glove.6B.300d.txt")
        super().__init__(name="AwA2")

    def process(self):
        transfor_binary_matrix, transfor_all_classes = transfor_matrix_binary()
        nodes_data = np.array(transfor_all_classes)
        train_nodes_data = get_train_classes()
        train_mask = np.zeros_like(nodes_data, dtype=np.int)
        test_mask = np.zeros_like(nodes_data, dtype=np.int)
        for i in range(len(nodes_data)):
            if nodes_data[i] in train_nodes_data:
                train_mask[i] = 1
            else:
                test_mask[i] = 1
        edge_data = None
        node_labels = torch.from_numpy(np.array([x for x in range(len(nodes_data))]))
        node_labels = CUDA(node_labels)
        # generate from glove
        # node_features = np.stack([self.glove[x].to_numpy() for x in node_labels])
        node_features = torch.from_numpy(transfor_binary_matrix)
        node_features = CUDA(node_features)
        edge_features = None
        edges_src, edges_dst = build_transform_edges()
        g = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        reverse_g = dgl.add_reverse_edges(g)
        # reverse_g_cuda = reverse_g.to("cuda:0")

        weight = torch.from_numpy(np.array([1 for _ in range(len(edges_dst) * 2)]))
        weight = CUDA(weight)
        train_mask = CUDA(torch.from_numpy(train_mask) > 0)
        test_mask = CUDA(torch.from_numpy(test_mask) > 0)

        # self.graph = reverse_g_cuda
        self.graph = reverse_g
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = weight
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1




class AwA2Conv(nn.Module):
    def __init__(self, num_feature, num_output):
        super(AwA2Conv, self).__init__()
        self.conv1 = GraphConv(num_feature, 512)
        self.conv2 = GraphConv(512, 1024)
        self.conv3 = GraphConv(1024, num_output)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        g.ndata["h"] = h
        return h
        # return dgl.mean_nodes(g, "h")

def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)

def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


def train(max_epochs):
    writer = SummaryWriter(log_dir="./log/zsl_graph/")
    dataset = AwA2GraphDataset()
    g = dataset[0]

    dim_label_feat = g.ndata["feat"].shape[1]
    dim_res50fc_feat = 2049
    dim_label = 50
    gcn_model = AwA2Conv(dim_label_feat, dim_res50fc_feat)
    gcn_model = CUDA(gcn_model)
    n_train = 40
    tlist = list(range(len(n_train)))
    random.shuffle(tlist)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=0.0005)

    gcn_model.train()

    for epoch in range(max_epochs):

        gcn_outputs = gcn_model(g, g.ndata["feat"])
        loss = mask_l2_loss(gcn_outputs, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_vectors = gcn_model(g, g.ndata["feat"])
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()

        print('epoch {}, train_loss={:.4f}'.format(epoch, train_loss))


        # histogram graph for parameters of networks
        for name, param in gcn_model.named_parameters():
            writer.add_histogram(
                name, param.clone().cpu().data.numpy(), max_epochs
            )

        if epoch % 5 == 0:
            writer.add_scalar("Loss", train_loss.item(), epoch)
            sys.stdout.flush()

    # save model
    torch.save(gcn_model.state_dict(), "models/{}".format("fc_gcn_model.bin"))


def test(output_filename):
    dataset = AwA2GraphDataset()
    g = dataset[0]

    dim_label_feat = g.ndata["feat"].shape[1]
    dim_res50_feat = 2048
    dim_label = 50
    gcn_model = AwA2Conv(dim_label_feat, dim_res50_feat)
    gcn_model = CUDA(gcn_model)
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
    train(lr=0.001, batch_size=24, epochs=300)
    # test(output_filename="zsl-graph-test.txt")

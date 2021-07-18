import dgl
import numpy as np
import torch
from dgl.data import DGLDataset

from build_graph import build_edges_on_predicates_above_average
from glove import GloVe
from utils import (
    CUDA,
    get_all_classes,
    get_predicate_binary_mat,
    get_test_classes,
    get_train_classes,
)


class AwA2GraphDataset(DGLDataset):
    def __init__(self):
        # self.glove = GloVe("materials/glove.6B.300d.txt")
        super().__init__(name="AwA2")

    def process(self):
        nodes_data = get_all_classes()
        train_nodes_data = get_train_classes()
        train_mask = np.zeros_like(nodes_data, dtype=np.int)
        test_mask = np.zeros_like(nodes_data, dtype=np.int)
        for i in range(len(nodes_data)):
            if nodes_data[i] in train_nodes_data:
                train_mask[i] = 1
            else:
                test_mask[i] = 1
        edge_data = None
        # node_labels = torch.from_numpy(
        #     np.array([x.replace("+", " ") for x in nodes_data])
        # )
        node_labels = torch.from_numpy(np.array([x for x in range(len(nodes_data))]))
        # generate from glove
        # node_features = np.stack([self.glove[x].to_numpy() for x in node_labels])
        node_features = torch.from_numpy(get_predicate_binary_mat())
        edge_features = None
        edges_src, edges_dst = build_edges_on_predicates_above_average()

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = torch.from_numpy(
            np.array([1 for _ in range(len(edges_dst))])
        )
        self.graph.ndata["train_mask"] = torch.from_numpy(train_mask) > 0
        self.graph.ndata["test_mask"] = torch.from_numpy(test_mask) > 0

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1

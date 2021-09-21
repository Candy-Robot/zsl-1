from os import name
import dgl
import networkx as nx
import torch
from utils import *


def build_graph_on_predicates_brute_force():
    """Brute force build graph based on predicates information directly."""
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = get_predicate_binary_mat()

    class_from = []
    class_end = []
    for i in range(predicates_dim):
        same_predicates = []
        for j in range(class_dim):
            if predicate_matrix_binary[j, i] == 1:
                # have same predicate
                same_predicates.append(j)
        # connect classes
        l = len(same_predicates)
        for j in range(l):
            for k in range(j, l):
                class_from.append(same_predicates[j])
                class_end.append(same_predicates[k])

    class_from, class_end = torch.tensor(class_from), torch.tensor(class_end)

    g = dgl.graph((class_from, class_end))
    return g


def build_edges_on_predicates_above_average():
    """Build graph based on predicates.
    When the number of predicates between two class above the average level,
    there's an edge between them.
    """
    # average same predicates between two class is 17.5
    avg_same_predicate = 17.5
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = get_predicate_binary_mat()

    edges = []

    for i in range(class_dim):
        for j in range(i, class_dim):
            num_same_predicate = 0
            for k in range(predicates_dim):
                if predicate_matrix_binary[i, k] == predicate_matrix_binary[j, k] == 1:
                    num_same_predicate += 1
            if num_same_predicate > avg_same_predicate:
                edges.append((i, j))

    # avg_same_predicate = num_same_predicate / (class_dim * (class_dim - 1) / 2)
    src_ids = torch.tensor([x[0] for x in edges])
    dst_ids = torch.tensor([x[1] for x in edges])
    return src_ids, dst_ids


def draw_graph(graph):
    """Draw graph."""
    nx_G = graph.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[0.7, 0.7, 0.7]])

def draw_graph_node(graph):
    """Draw graph single node"""
    nx_G = graph.to_networkx().to_undirected()
    nx_G
    

def build_transform_edges():
    transfor_binary_matrix, transfor_all_classes = transfor_matrix_binary()
    avg_same_predicate = 17.5
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = transfor_binary_matrix

    edges = []

    for i in range(class_dim):
        for j in range(i, class_dim):
            num_same_predicate = 0
            for k in range(predicates_dim):
                if predicate_matrix_binary[i, k] == predicate_matrix_binary[j, k] == 1:
                    num_same_predicate += 1
            if num_same_predicate > avg_same_predicate:
                edges.append((i, j))

    # avg_same_predicate = num_same_predicate / (class_dim * (class_dim - 1) / 2)
    src_ids = torch.tensor([x[0] for x in edges])
    dst_ids = torch.tensor([x[1] for x in edges])
    return src_ids, dst_ids


def build_dense_edges_on_predicates_less():
    transfor_binary_matrix, transfor_all_classes = transfor_matrix_binary()
    avg_same_predicate = 10
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = transfor_binary_matrix
    edges = []

    for i in range(class_dim):
        for j in range(i, class_dim):
            num_same_predicate = 0
            for k in range(predicates_dim):
                if predicate_matrix_binary[i, k] == predicate_matrix_binary[j, k] == 1:
                    num_same_predicate += 1
            if num_same_predicate > avg_same_predicate:
                edges.append((i, j))

    # avg_same_predicate = num_same_predicate / (class_dim * (class_dim - 1) / 2)
    src_ids = torch.tensor([x[0] for x in edges])
    dst_ids = torch.tensor([x[1] for x in edges])
    return src_ids, dst_ids

def build_edges_without_unseen_classes_edges():
    transfor_binary_matrix, transfor_all_classes = transfor_matrix_binary()
    avg_same_predicate = 17.5
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = transfor_binary_matrix
    
    test_classes = get_test_classes()
    class_to_index = mapping_class_to_index()
    test_index = []
    for i in range(len(test_classes)):
        test_index.append(class_to_index[test_classes[i]])
    edges = []

    for i in range(class_dim):
        for j in range(i, class_dim):
            num_same_predicate = 0
            for k in range(predicates_dim):
                if predicate_matrix_binary[i, k] == predicate_matrix_binary[j, k] == 1:
                    num_same_predicate += 1
            if num_same_predicate > avg_same_predicate:
                if i not in test_index and j not in test_index:
                    edges.append((i, j))    

    # avg_same_predicate = num_same_predicate / (class_dim * (class_dim - 1) / 2)
    src_ids = torch.tensor([x[0] for x in edges])
    dst_ids = torch.tensor([x[1] for x in edges])
    return src_ids, dst_ids

def build_all_edges_without_unseen_edges():
    transfor_binary_matrix, transfor_all_classes = transfor_matrix_binary()
    avg_same_predicate = 1
    predicates_dim = 85
    class_dim = 50
    predicate_matrix_binary = transfor_binary_matrix
    
    test_classes = get_test_classes()
    class_to_index = mapping_class_to_index()
    test_index = []
    for i in range(len(test_classes)):
        test_index.append(class_to_index[test_classes[i]])
    edges = []
    success_case = 0

    for i in range(class_dim):
        for j in range(i, class_dim):
            if i not in test_index and j not in test_index:
                for k in range(predicates_dim): 
                    if predicate_matrix_binary[i, k] == predicate_matrix_binary[j, k] == 1:
                        edges.append((i, j))
                        success_case += 1
                        break

                    



    src_ids = torch.tensor([x[0] for x in edges])
    dst_ids = torch.tensor([x[1] for x in edges])
    return src_ids, dst_ids


if __name__ == "__main__":
    src_ids, dst_ids = build_all_edges_without_unseen_edges()
    g = dgl.graph((src_ids, dst_ids))
    draw_graph(g)
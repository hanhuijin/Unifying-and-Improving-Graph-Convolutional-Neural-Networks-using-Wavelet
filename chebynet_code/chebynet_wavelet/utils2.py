import random
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx


def load_data2(dataset_str, max_subgraph=False):
    # features = []
    # labels = []
    # adj_lists = {}    # graph的格式
    # adj = []
    if dataset_str == "cora":
        if max_subgraph == "max_subgraph_cora":
            features, labels, adj = load_max_subgraph_cora()
        else:
            features, labels, adj_lists, adj = load_cora()
    # elif dataset_str == "citeseer":
    #     features, labels, adj_lists, adj = load_citeseer()
    else:
        raise Exception("咱不支持该数据集")
    labels = to_categorical(labels)     # 分类格式由类别号转换为one-hot方式
    class_num = labels.shape[1]     # 类别数量
    total_num = labels.shape[0]     # 数据集数据量
    # train_num = class_num * 20
    val_num = 500
    test_num = 1000
    all_index = list(range(total_num))
    shuffle = all_index.copy()
    random.shuffle(shuffle)       # 为实现数据集划分的随机性，将数据的索引打乱，后续从乱序索引中从前开始取数即可
    class_num_flag = [0] * class_num
    train_index = []
    for j in range(class_num):      # 每一类收集20个训练集数据索引
        for i in shuffle:
            if class_num_flag[j] < 20:
                if labels[i][j]:
                    train_index.append(i)
                    all_index.remove(i)     # 被收集到训练集后从所有索引集中剔除
                    class_num_flag[j] += 1
                    continue
            else:       # 若该类数据已有20个，则结束该类数据收集过程
                break

    print('train_index: ', train_index)
    print('len of train_index', len(train_index))
    print('len of all_index: ', len(all_index))
    # 从剩下的索引中取验证集和测试集索引
    shuffle_2 = all_index.copy()
    random.shuffle(shuffle_2)
    val_index = shuffle_2[:val_num]     # 验证集数据索引
    test_index = shuffle_2[val_num:val_num+test_num]        # 测试集数据索引
    print('test_index: ', test_index)
    print('len of test_index', len(test_index))

    # 制作掩膜：在所有索引上盖住对应索引（置为True）
    train_mask = sample_mask(train_index, total_num)
    val_mask = sample_mask(val_index, total_num)
    test_mask = sample_mask(test_index, total_num)

    # 各数据集对应标签基底
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    # 在各数据集对应标签基底上保存对应的标签数据
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return labels, adj, csr_matrix(features).tolil(), y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_cora(num_sample=5):
    # hardcoded for simplicity...
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype="float32")
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("data2/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("data2/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in node_map or info[1] not in node_map:
                print(info[0], info[1])
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    # adj = np.zeros((num_nodes, num_nodes), dtype='int32')
    adj_lists = defaultdict(list, ((k, list(v)) for k, v in adj_lists.items()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj_lists))
    # for paper1, nodes_list in adj_lists.items():
    #     for paper2 in nodes_list:
    #         adj[paper1, paper2] = 1.0
    #         adj[paper2, paper1] = 1.0

    return feat_data, labels, adj_lists, adj

def load_max_subgraph_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats), dtype="float32")
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("data2/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("data2/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in node_map or info[1] not in node_map:
                print(info[0], info[1])
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    # adj = np.zeros((num_nodes, num_nodes), dtype='int32')
    adj_lists = defaultdict(list, ((k, list(v)) for k, v in adj_lists.items()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj_lists))
    G = nx.Graph(adj)
    largest_graph = max(nx.connected_components(G), key=len)
    largest_graph_index = list(largest_graph)
    sub_adj = adj[largest_graph_index][:, largest_graph_index]
    sub_feat_data = feat_data[largest_graph_index]
    sub_labels = labels[largest_graph_index]
    return sub_feat_data, sub_labels, sub_adj

def load_pubmed(num_sample=10):
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("./pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("./pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    adj = np.zeros((num_nodes, num_nodes), dtype='float32')
    for paper1, nodes_list in adj_lists.items():
        for paper2 in nodes_list:
            adj[paper1, paper2] = 1.0
            adj[paper2, paper1] = 1.0
    return feat_data, labels, adj_lists, adj


def load_citeseer(num_sample=5):
    # hardcoded for simplicity...
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("./data2/citeseer/citeseer.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("./data2/citeseer/citeseer.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in node_map or info[1] not in node_map:
                print(info[0], info[1])
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    adj = np.zeros((num_nodes, num_nodes), dtype='float32')
    for paper1, nodes_list in adj_lists.items():
        for paper2 in nodes_list:
            adj[paper1, paper2] = 1.0
            adj[paper2, paper1] = 1.0
    return feat_data, labels, adj_lists, adj


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype="int32")
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


if __name__ == "__main__":
    import networkx as nx

    feat_data, labels, adj_lists, adj= load_cora(5)
    print("labels:", labels)
    labels_one_hot = to_categorical(labels)
    print("labels_one_hot:", labels_one_hot)
    # feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_pubmed(10)
    # feat_data, labels, adj_lists, adj = load_citeseer(1)
    G = nx.Graph(adj)
    # for i in range(len(adj)):
    #     for j in range(len(adj[0])):
    #         if adj[i][j]==1.0:
    #             G.add_edge(i, j)
    # print(nx.clustering(G))
    # print('average_clustering',nx.average_clustering(G))
    num = 0
    graph_list = [G.subgraph(c) for c in nx.connected_components(G)]
    for C in graph_list:
        num = num +1
        # print('average_shortest_path_length', nx.average_shortest_path_length(C))
        # print(C)
        C_adj = nx.adjacency_matrix(C)
        print(len(C.nodes))
        # print(len(C.edges))
    print('num_of_subgraph', num)
    print('transitivity', nx.transitivity(G))
    import networkx.algorithms.community as nx_comm

    # partition = nx_comm.louvain_communities(G)
    # print('nx_comm.louvain_communities', partition)

    # # drawing待探索
    # size = float(len(set(partition)))
    # pos = nx.spring_layout(G)
    # count = 0.
    # for com in set(partition):
    #     count = count + 1.
    #     list_nodes = [nodes for nodes in partition.keys()
    #                   if partition[nodes] == com]
    #     nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
    #                            node_color=str(count / size))
    #
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()

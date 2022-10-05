import random

from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from weighting_func import laplacian, fourier, weight_wavelet, weight_wavelet_inverse
import tensorflow.compat.v1 as tf
import warnings

warnings.filterwarnings("ignore")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# 数据的读取，这个预处理是把训练集（其中一部分带有标签），测试集，标签的位置，对应的掩码训练标签等返回。
def load_data(dataset_str, alldata=True, random_train=True, max_subgraph = False):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)  # 转化为tuple
    # 以Cora数据集为例
    # x是一个测试数据集的稀疏矩阵,记住1的位置,140个实例,每个实例的特征向量维度是1433  (140,1433)
    # y是测试数据集的标签向量,7分类，140个实例 (140,7)
    # tx是训练数据集的稀疏矩阵，1000个实例,每个实例的特征向量维度是1433  (1000,1433)
    # ty是训练数据集的标签向量，7分类，1000个实例 (1000,7)
    # allx是一个验证数据的稀疏矩阵,1708个实例,每个实例的特征向量维度是1433  (1708,1433)
    # ally是验证数据集的标签向量,7分类，1708个实例 (1708,7)
    # graph是一个字典，大图总共2708个节点

    # 测试数据集的索引乱序版
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # 从小到大排序,如[1707,1708,1709,...]
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        # 转化成LIL格式的稀疏矩阵,tx_extended.shape=(1015,1433)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # test_idx_range-min(test_idx_range):列表中每个元素都减去min(test_idx_range)，即将test_idx_range列表中的index值变为从0开始编号
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if max_subgraph:
        G = nx.Graph(adj)
        largest_graph = max(nx.connected_components(G), key=len)
        largest_graph_index = list(largest_graph)
        adj = adj[largest_graph_index][:, largest_graph_index]
        features = features[largest_graph_index]
        labels = labels[largest_graph_index]

    total_num = len(labels)
    train_num = 140
    val_num = 500
    test_num = 1000
    # 给所有索引列表随机打乱顺序，取其中不重复的数分别作为测试集、训练集、验证集
    shuffle = list(range(total_num))
    random.shuffle(shuffle)
    idx_train = shuffle[0:train_num]
    idx_val = shuffle[train_num:val_num+train_num]
    idx_test = shuffle[val_num+train_num:test_num+train_num+val_num]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)

    # if alldata:
    #     features = sp.vstack((allx, tx)).tolil()
    #     labels = np.vstack((ally, ty))
    #     num = labels.shape[0]
    #     idx_train = range(num / 5 * 3)
    #     idx_val = range(num / 5 * 3, num / 5 * 4)
    #     idx_test = range(num / 5 * 4, num)
    if alldata:
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
        num = labels.shape[0]
        shuffle = list(range(num))
        random.shuffle(shuffle)
        s1 = int(num / 5 * 1)
        s2 = int(num / 5 * 3)
        train_num = s1
        val_num = s2 - s1
        test_num = num - s2
        idx_train = shuffle[:s1]
        idx_val = shuffle[s1:s2]
        idx_test = shuffle[s2:num]

    if random_train:
        class_num = labels.shape[1]
        class_num_flag = [0]*class_num
        all_index = [i for i in range(total_num)]
        train_index = []
        for j in range(class_num):
            for i in shuffle:
                if class_num_flag[j] < 20:
                    if labels[i][j]:
                        class_num_flag[j] += 1
                        train_index.append(i)
                        all_index.remove(i)
                        continue
                else:
                    break
        print('train_index: ', train_index)
        print('len of train_index', len(train_index))
        print('len of all_index: ', len(all_index))
        idx_train = train_index
        # idx_val = all_index[:val_num]
        # idx_test = all_index[val_num:test_num+val_num]
        shuffle_2 = all_index.copy()
        random.shuffle(shuffle_2)
        idx_val = shuffle_2[:val_num]  # 验证集数据索引
        idx_test = shuffle_2[val_num:val_num + test_num]  # 测试集数据索引
        print('idx_test: ', idx_test)
        print('len of idx_test', len(idx_test))

    # 训练mask：idx_train=[0,140)范围的是True，后面的是False
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    # 替换了true位置
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


# 将稀疏矩sparse_mx阵转换成tuple格式并返回
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# 处理特征:特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加
    rowsum = np.array(features.sum(1))
    # print rowsum
    r_inv = np.power(rowsum, -1).flatten()
    # np.isinf(ndarray)返回一个判断是否是NaN的bool型数组
    r_inv[np.isinf(r_inv)] = 0.
    # sp.diags创建一个对角稀疏矩阵
    r_mat_inv = sp.diags(r_inv, 0)
    # dot矩阵乘法
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


# 邻接矩阵adj对称归一化并返回coo存储模式
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回元组格式
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # 加上自环，再对称归一化
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return adj_normalized
    return sparse_to_tuple(adj_normalized)


# 构建输入字典并返回
# labels和labels_mask传入的是具体的值，例如
# labels=y_train,labels_mask=train_mask；
# labels=y_val,labels_mask=val_mask；
# labels=y_test,labels_mask=test_mask；

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    # 由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# 切比雪夫多项式近似:计算K阶的切比雪夫近似矩阵
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized   # L
    largest_eigval, _ = eigsh(laplacian, 1, which='LM') # lambda
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0]) # Lba = 2A/lambda{max} - I

    # 切比雪夫多项式近似:计算K阶的切比雪夫近似矩阵
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    # 依据公式 T_n(x) = 2xT_n(x) - T_{n-1}(x) 构造递归程序，计算T_2 -> T_k
    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def wavelet_basis(dataset, adj, s, laplacian_normalize, sparse_ness, threshold, weight_normalize, wave_normalize_type):
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset, L)
    Weight = weight_wavelet(s, lamb, U)
    inverse_Weight = weight_wavelet_inverse(s, lamb, U)
    del U, lamb

    if sparse_ness:
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0
        if wave_normalize_type == 'norm1':
            min_Weight = np.min(Weight)
            max_Weight = np.max(Weight)
            Weight = 1 + np.divide(Weight - min_Weight, max_Weight - min_Weight)
            min_inverse_Weight = np.min(inverse_Weight)
            max_inverse_Weight = np.max(inverse_Weight)
            inverse_Weight = 1 + np.divide(inverse_Weight - min_inverse_Weight, max_inverse_Weight - min_inverse_Weight)
    # print len(np.nonzero(Weight)[0])

    if weight_normalize:
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    # print Weight
    t_k = [inverse_Weight, Weight]
    return sparse_to_tuple(t_k)


def wave_cheby_basis(dataset, adj, s, laplacian_normalize, sparse_ness, threshold, weight_normalize, wave_normalize_type):
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset, L)
    Weight = weight_wavelet(s, lamb, U) # psai-1
    inverse_Weight = weight_wavelet_inverse(s, lamb, U)    # psai
    del lamb

    if sparse_ness:
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0
        if wave_normalize_type == 'norm1':
            min_Weight = np.min(Weight)
            max_Weight = np.max(Weight)
            Weight = 1 + np.divide(Weight - min_Weight, max_Weight - min_Weight)
            min_inverse_Weight = np.min(inverse_Weight)
            max_inverse_Weight = np.max(inverse_Weight)
            inverse_Weight = 1 + np.divide(inverse_Weight - min_inverse_Weight, max_inverse_Weight - min_inverse_Weight)
    # print len(np.nonzero(Weight)[0])

    if weight_normalize:
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    # print Weight
    U = sp.csr_matrix(U)
    t_k = [U, inverse_Weight, Weight]
    return sparse_to_tuple(t_k)


def wave_cheby_basis_shrink(dataset, adj, s, laplacian_normalize, sparse_ness, threshold, weight_normalize, wave_normalize_type):
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset, L)
    Weight = weight_wavelet(s, lamb, U) # psai-1
    inverse_Weight = weight_wavelet_inverse(s, lamb, U)    # psai
    del lamb

    if sparse_ness:
        Weight = np.abs(Weight)
        inverse_Weight = np.abs(inverse_Weight)
        Weight[Weight < threshold] = 0.0
        Weight[Weight >= threshold] = Weight[Weight >= threshold] - threshold
        inverse_Weight[inverse_Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight >= threshold] = inverse_Weight[inverse_Weight >= threshold] - threshold
        if wave_normalize_type == 'norm1':
            min_Weight = np.min(Weight)
            max_Weight = np.max(Weight)
            Weight = 1 + np.divide(Weight - min_Weight, max_Weight - min_Weight)
            min_inverse_Weight = np.min(inverse_Weight)
            max_inverse_Weight = np.max(inverse_Weight)
            inverse_Weight = 1 + np.divide(inverse_Weight - min_inverse_Weight, max_inverse_Weight - min_inverse_Weight)
    # print len(np.nonzero(Weight)[0])

    if weight_normalize:
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    # print Weight
    U = sp.csr_matrix(U)
    t_k = [U, inverse_Weight, Weight]
    return sparse_to_tuple(t_k)


def spectral_basis(dataset, adj, s, laplacian_normalize, sparse_ness, threshold, weight_normalize):
    from weighting_func import laplacian, fourier, weight_wavelet, weight_wavelet_inverse
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset, L)

    U = sp.csr_matrix(U)
    # U_transpose = sp.csr_matrix(np.transpose(U))
    t_k = [U]
    return sparse_to_tuple(t_k)

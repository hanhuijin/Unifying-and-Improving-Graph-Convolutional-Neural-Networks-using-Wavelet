import pickle as cp
import inspect
import numpy as np
import os
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

def _parse_cora_features_labels():
    path = "%s/../data/cora/" % (current_dir,)

    id2index = {}

    label2index = {
        'Case_Based': 0,
        'Genetic_Algorithms': 1,
        'Neural_Networks': 2,
        'Probabilistic_Methods': 3,
        'Reinforcement_Learning': 4,
        'Rule_Learning': 5,
        'Theory': 6
    }

    features = []
    labels = []

    with open(path + 'cora.content', 'r') as f:
        i = 0
        for line in f.readlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(label)

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.asarray(labels, dtype='int32')

    return features, labels, id2index


def parse_cora(tau):
    path = "%s/../data/cora/" % (current_dir,)

    features, labels, id2index = _parse_cora_features_labels()

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')
    wave = np.zeros((n_papers, n_papers), dtype='float32')
    wave_new = np.zeros((n_papers, n_papers), dtype='float32')
    #————————————————————————————————————————————————-------------------------------------------
    from collections import defaultdict
    chev_order = 3
    thre = 0.001
    # tau = [5, ]
    num_sample = 10
    num_nodes = 2708
    num_feats = 1433
    import pickle as pkl
    import scipy.sparse as sparse
    from pygsp import graphs, filters, plotting, utils
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)  # wavelet 基过滤后的列表
    with open(path + 'cora.cites', 'r') as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = id2index[info[0]]
            paper2 = id2index[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    wave_file = "cora" + "chev_" + str(chev_order) + "sample_" + str(num_sample) + ".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes, num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1

        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau)  # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j] > thre:
                    ls.append((j, s[i][j]))
            if len(ls) < num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x: x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
    value_of_s_inwavelist = []
    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1 != paper2:
                value_of_s_inwavelist.append(s[paper1, paper2] - thre)
    max_of_s = np.max(value_of_s_inwavelist)
    min_of_s = np.min(value_of_s_inwavelist)
    mean_of_s = np.mean(value_of_s_inwavelist)  # np.mean(value_of_s_inwavelist)#np.sum(s[:,:])/(len(wave_lists))
    std_of_s = np.std(value_of_s_inwavelist)
    median_of_s = np.median(value_of_s_inwavelist)
    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:

            wave[paper1, paper2] = 1.0
            wave[paper2, paper1] = 1.0

    for paper1, nodes_list in adj_lists.items():
        for paper2 in nodes_list:
            adj[paper1, paper2] = 1.0
            adj[paper2, paper1] = 1.0



    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1 == paper2:
                wave_new[paper1, paper2] = 1.0
            else:
                wave_new[paper1, paper2] = 1.0+(s[paper1, paper2] - thre - min_of_s) / (
                            max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
                wave_new[paper2, paper1] = 1.0+(s[paper2, paper1] - thre - min_of_s) / (
                            max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）

    #——————————————————————————————————————————————————————————————
    #改后注释掉了
    # with open(path + 'cora.cites', 'r') as f:
    #     for line in f.readlines():
    #         items = line.strip().split('\t')
    #         adj[id2index[items[0]], id2index[items[1]]] = 1.0
    #         # undirected
    #         adj[id2index[items[1]], id2index[items[0]]] = 1.0
    #___________________________________largest graph
    import networkx as nx
    G = nx.Graph(adj)
    largest_graph = list(max(nx.connected_components(G), key=len))
    adj = adj[largest_graph][:, largest_graph]
    wave = wave[largest_graph][:, largest_graph]
    wave_new = wave_new[largest_graph][:, largest_graph]
    features = features[largest_graph]
    labels = labels[largest_graph]
    num_nodes = len(largest_graph)
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] != 0:
                adj_lists[i].add(j)
            if wave[i][j] != 0:
                wave_lists[i].add(j)

    # ___________________________________largest graph


    return adj.astype('float32'),wave.astype('float32'),wave_new.astype('float32'), features.astype('float32'), labels.astype('int32')


def parse_cora_sparse():
    path = "%s/../data/cora/" % (current_dir,)

    features, labels, id2index = _parse_cora_features_labels()

    n_papers = len(id2index)
    graph = nx.Graph()

    with open(path + 'cora.cites', 'r') as f:
        for line in f.xreadlines():
            items = line.strip().split('\t')

            tail = id2index[items[0]]
            head = id2index[items[1]]

            graph.add_edge(head, tail)

    adj = nx.to_scipy_sparse_matrix(graph, format='csr')

    return adj.astype('float32'), features.astype('float32'), labels.astype('int32')

#_____________________________________________________________________________________________________________
#修改添加
def _parse_citeseer_features_labels():
    path = "%s/../data/citeseer/" % (current_dir,)

    id2index = {}

    label2index = {
        'Agents': 0,
        'IR': 1,
        'DB': 2,
        'AI': 3,
        'HCI': 4,
        'ML': 5
    }

    features = []
    labels = []

    with open(path + 'citeseer.content', 'r') as f:
        i = 0
        for line in f.readlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = np.zeros(len(label2index))
            label[label2index[items[-1]]] = 1
            labels.append(label)

            # parse features
            features.append([int(x) for x in items[1:-1]])


            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32')
    labels = np.asarray(labels, dtype='int32')

    return features, labels, id2index


def parse_citeseer(tau):
    path = "%s/../data/citeseer/" % (current_dir,)

    features, labels, id2index = _parse_citeseer_features_labels()

    n_papers = len(id2index)

    adj = np.zeros((n_papers, n_papers), dtype='float32')
    wave = np.zeros((n_papers, n_papers), dtype='float32')
    wave_new = np.zeros((n_papers, n_papers), dtype='float32')
    #————————————————————————————————————————————————-------------------------------------------
    from collections import defaultdict
    chev_order = 3
    thre = 0.001
    # tau = [5, ]
    num_sample = 2110
    num_nodes = 3312
    num_feats = 3703
    import pickle as pkl
    import scipy.sparse as sparse
    from pygsp import graphs, filters, plotting, utils
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)  # wavelet 基过滤后的列表
    with open(path + 'citeseer.cites', 'r') as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in id2index or info[1] not in id2index:
                print(info[0],info[1])
                continue
            paper1 = id2index[info[0]]
            paper2 = id2index[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    wave_file = "citeseer" + "chev_" + str(chev_order) + "sample_" + str(num_sample) + ".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes, num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1

        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau)  # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j] > thre:
                    ls.append((j, s[i][j]))
            if len(ls) < num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x: x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
    value_of_s_inwavelist=[]
    for paper1,nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1!=paper2:
                value_of_s_inwavelist.append(s[paper1, paper2]-thre)

    max_of_s=np.max(value_of_s_inwavelist)
    min_of_s=np.min(value_of_s_inwavelist)
    mean_of_s=np.mean(value_of_s_inwavelist)#np.mean(value_of_s_inwavelist)#np.sum(s[:,:])/(len(wave_lists))
    std_of_s=np.std(value_of_s_inwavelist)
    median_of_s=np.median(value_of_s_inwavelist)
    for paper1,nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1==paper2:
                wave_new[paper1,paper2]=1.0
            else:
                wave_new[paper1,paper2]=1.0+(s[paper1,paper2]-thre-min_of_s)/(max_of_s-min_of_s)#实验一下老师新说的，看会不会有效果（原来为1）
                wave_new[paper2,paper1]=1.0+(s[paper2,paper1]-thre-min_of_s)/(max_of_s-min_of_s)#实验一下老师新说的，看会不会有效果（原来为1）
    # wave_new=(wave_new-np.mean(s[:,:]-thre))/np.std(s[:,:]-thre)
    for paper1,nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            wave[paper1,paper2]=1.0
            wave[paper2,paper1]=1.0

    for paper1,nodes_list in adj_lists.items():
        for paper2 in nodes_list:
            adj[paper1,paper2]=1.0
            adj[paper2,paper1]=1.0
    #——————————————————————————————————————————————————————————————
    #改后注释掉了
    # with open(path + 'citeseer.cites', 'r') as f:
    #     for line in f.readlines():
    #         items = line.strip().split('\t')
    #         if items[0] not in id2index or items[1] not in id2index:
    #             print(items[0],items[1])
    #             continue
    #
    #         adj[id2index[items[0]], id2index[items[1]]] = 1.0
    #         # undirected
    #         adj[id2index[items[1]], id2index[items[0]]] = 1.0

    #___________________________________largest graph
    import networkx as nx
    G = nx.Graph(adj)
    largest_graph = list(max(nx.connected_components(G), key=len))
    adj = adj[largest_graph][:, largest_graph]
    wave = wave[largest_graph][:, largest_graph]
    wave_new = wave_new[largest_graph][:, largest_graph]
    features = features[largest_graph]
    labels = labels[largest_graph]
    num_nodes = len(largest_graph)
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] != 0:
                adj_lists[i].add(j)
            if wave[i][j] != 0:
                wave_lists[i].add(j)

    # ___________________________________largest graph

    return adj.astype('float32'),wave.astype('float32'),wave_new.astype('float32'), features.astype('float32'), labels.astype('int32')
#_________________________________________________________________________________________________________________________


def parse_pubmed(tau,num_sample=10):
    # hardcoded for simplicity...
    from collections import defaultdict
    import pickle as pkl
    import scipy.sparse as sparse
    from pygsp import graphs, filters, plotting, utils
    chev_order = 3
    thre = 0.001
    # tau = [5, ]
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = []  # np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("../data/pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            label = np.zeros(3)
            label[int(info[1].split("=")[1]) - 1] = 1
            labels.append(label)
            #             labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)
    with open("../data/pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    wave_file = "pubmed" + "chev_" + str(chev_order) + "sample_" + str(num_sample) + ".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes, num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1

        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau)  # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j] > thre:
                    ls.append((j, s[i][j]))
            if len(ls) < num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x: x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)

    n_papers = len(node_map)
    adj = np.zeros((n_papers, n_papers), dtype='float32')
    wave = np.zeros((n_papers, n_papers), dtype='float32')
    wave_new=np.zeros((n_papers, n_papers), dtype='float32')
    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            wave[paper1, paper2] = 1.0
            wave[paper2, paper1] = 1.0
    for paper1, nodes_list in adj_lists.items():
        for paper2 in nodes_list:
            adj[paper1, paper2] = 1.0
            adj[paper2, paper1] = 1.0
    value_of_s_inwavelist = []
    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1 != paper2:
                value_of_s_inwavelist.append(s[paper1, paper2] - thre)
    max_of_s = np.max(value_of_s_inwavelist)
    min_of_s = np.min(value_of_s_inwavelist)
    mean_of_s = np.mean(value_of_s_inwavelist)  # np.mean(value_of_s_inwavelist)#np.sum(s[:,:])/(len(wave_lists))
    std_of_s = np.std(value_of_s_inwavelist)
    median_of_s = np.median(value_of_s_inwavelist)
    for paper1, nodes_list in wave_lists.items():
        for paper2 in nodes_list:
            if paper1 == paper2:
                wave_new[paper1, paper2] = 1.0
            else:
                wave_new[paper1, paper2] = 1.0+(s[paper1, paper2] - thre - min_of_s) / (
                            max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
                wave_new[paper2, paper1] = 1.0+(s[paper2, paper1] - thre - min_of_s) / (
                            max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
    labels = np.asarray(labels, dtype='int32')

    return adj.astype('float32'), wave.astype('float32'),wave_new.astype('float32'),feat_data.astype('float32'), labels.astype(
        'int32')  # feat_data, labels, adj_lists, wave_lists


def parse_graph_data(graph_name='nci1.graph'):
    path = "%s/../data/" % (current_dir,)

    if graph_name == 'nci1.graph':
        maxval = 37
        n_classes = 2
    elif graph_name == 'nci109.graph':
        maxval = 38
        n_classes = 2
    elif graph_name == 'mutag.graph':
        maxval = 7
        n_classes = 2
    elif graph_name == 'ptc.graph':
        maxval = 22
        n_classes = 2
    elif graph_name == 'enzymes.graph':
        maxval = 3
        n_classes = 6

    with open(path+graph_name,'r') as f:
        raw = cp.load(f)

        n_graphs = len(raw['graph'])

        A = []
        rX = []
        Y = []

        for i in range(n_graphs):
            # Set label
            class_label = raw['labels'][i]

            y = np.zeros((1, n_classes), dtype='int32')

            if n_classes == 2:
                if class_label == 1:
                    y[0,1] = 1
                else:
                    y[0,0] = 1
            else:
                y[0,class_label-1] = 1

            # Parse graph
            G = raw['graph'][i]

            n_nodes = len(G)

            a = np.zeros((n_nodes, n_nodes), dtype='float32')
            x = np.zeros((n_nodes, maxval), dtype='float32')

            for node, meta in G.iteritems():
                label = meta['label'][0] - 1
                x[node, label] = 1
                for neighbor in meta['neighbors']:
                    a[node, neighbor] = 1

            A.append(a)
            rX.append(x)
            Y.append(y)

    return A, rX, Y


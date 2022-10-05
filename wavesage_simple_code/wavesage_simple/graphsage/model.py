import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import scipy.sparse as sparse
from pygsp import graphs, filters, plotting, utils

import os
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
chev_order = 3
thre = 0.001
tau = [1.2, ]
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(4)

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        # test=torch.FloatTensor(num_classes, enc.embed_dim)

        # print(test)

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        # self.weight.data.fill_(0.1)
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
    def fit(self, train, valid,labels):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.005)
        # times = []
        es = 0
        best_acc = 0.0
        best_score=None
        for epoch in range(60):
            # optimizer.state_dict()['param_groups'][0]['lr'] *=(epoch/10)
            train_loss=0.0
            num_nodes=len(train)
            batch_size=20
            num_batch = num_nodes // batch_size
            for batch in range(num_batch):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, num_nodes)
                if start < end:
                    batch_nodes = train[start:end]
                    # random.shuffle(train)
                    # start_time = time.time()
                    optimizer.zero_grad()
                    loss = self.loss(batch_nodes,
                                  Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
                    loss.backward()
                    optimizer.step()
                    # end_time = time.time()
                    # times.append(end_time - start_time)
                    # print(batch, loss.item())
                    train_loss+=loss
            train_loss/=num_batch
            valid_loss=self.loss(valid,
                                  Variable(torch.LongTensor(labels[np.array(valid)])))
            print("Epoch %d mean training error: %.6f" % (epoch, train_loss))
            print("Epoch %d validation error: %.6f" % (epoch, valid_loss))
            val_acc=accuracy_score(torch.argmax(self.forward(valid),1),
                                  torch.LongTensor(labels[np.array(valid)]))
            # score=-valid_loss
            # if best_score is None:
            #     best_score=score
            #     es = 0
            #     best_acc=val_acc
            #     torch.save(self.state_dict(), "checkpoint.pt")
            # elif score<best_score:
            #     best_score = score
            #     es = 0
            #     torch.save(self.state_dict(), "checkpoint.pt")
            # else:
            #     es += 1
            #     optimizer.state_dict()['param_groups'][0]['lr'] /= 2
            #     # print("Counter {} of 5".format(es))
            #     if es > 4 :
            #         print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
            #         self.load_state_dict(torch.load('checkpoint.pt'))
            #         print('val_acc', accuracy_score(torch.argmax(self.forward(valid), 1),
            #                                         torch.LongTensor(labels[np.array(valid)])))
            #         break



            if val_acc >best_acc :
                best_acc = val_acc
                es = 0
                torch.save(self.state_dict(), "checkpoint.pt")
            else:
                es += 1
                optimizer.state_dict()['param_groups'][0]['lr'] /= 2
                # print("Counter {} of 5".format(es))
                if es > 4 and best_acc>0.7:
                    print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
                    self.load_state_dict(torch.load('checkpoint.pt'))
                    print('val_acc', accuracy_score(torch.argmax(self.forward(valid), 1),
                                                    torch.LongTensor(labels[np.array(valid)])))
                    break
            self.load_state_dict(torch.load('checkpoint.pt'))


def load_cora(num_sample = 10):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set) # wavelet 基过滤后的列表
    with open("../cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    wave_file = "cora"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))

        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
##
#______________________________________________________________________________________plus
    adj = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave_new = np.zeros((num_nodes, num_nodes), dtype='float32')
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
                wave_new[paper1, paper2] = 1.0 + (s[paper1, paper2] - thre - min_of_s) / (
                        max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
                wave_new[paper2, paper1] = 1.0 + (s[paper2, paper1] - thre - min_of_s) / (
                        max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
#_______________________________________________________________________________________________________plus
##
    return feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new
def run_cora_method(sample_method, train, val, test,feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new, num_nodes, gcn=True):
    features = nn.Embedding(num_nodes, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    # agg1 = MeanAggregator(features, cuda=True)
    if sample_method == "adj":
        agg1 = MeanAggregator(features, adj, cuda=True)
        enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg1 = MeanAggregator(features, wave, cuda=True)
        enc1 = Encoder(features, 1433, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg1 = MeanAggregator(features, wave_new, cuda=True)
        enc1 = Encoder(features, 1433, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    # agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    if sample_method == "adj":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), adj, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave_new, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
    graphsage.fit(train, val, labels)
    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    rec = ""
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:" + str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print ("Average batch time:", np.mean(times))
    # rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res.txt", "cora" + sample_method + str(gcn) + rec)
    return f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
def run_cora(sample_method,  train, val, test, gcn=True):
    # tau = [3,]
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_cora(5)
    #___________________________________largest graph
    import networkx as nx
    G = nx.Graph(adj)
    largest_graph = list(max(nx.connected_components(G), key=len))
    adj = adj[largest_graph][:, largest_graph]
    wave = wave[largest_graph][:, largest_graph]
    wave_new = wave_new[largest_graph][:, largest_graph]
    feat_data = feat_data[largest_graph]
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
    method = ['wave_new', 'wave', 'adj']
    test_acc_wavenew = 0.0
    test_acc_wave = 0.0
    test_acc_adj = 0.0

    test_acc_wavenew = run_cora_method('wave_new', train, val, test, feat_data, labels, adj_lists, wave_lists, adj,
                                         wave,
                                         wave_new, num_nodes, gcn=gcn)
    print('wave_new', gcn, test_acc_wavenew)
    test_acc_wave = run_cora_method('wave', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                                      wave_new, num_nodes, gcn=gcn)
    print('wave', gcn, test_acc_wave)
    test_acc_adj = run_cora_method('adj', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                                     wave_new, num_nodes, gcn=gcn)
    print('adj', gcn, test_acc_adj)
    return test_acc_wavenew, test_acc_wave, test_acc_adj

def load_pubmed(num_sample = 10):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("../pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)
    with open("../pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    
    wave_file = "pubmed"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
##
#——————————————————————————————————————————————————————————————————————————————————————————————————————plus
    adj = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave_new = np.zeros((num_nodes, num_nodes), dtype='float32')
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
                wave_new[paper1, paper2] = 1.0 + (s[paper1, paper2] - thre - min_of_s) / (
                        max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
                wave_new[paper2, paper1] = 1.0 + (s[paper2, paper1] - thre - min_of_s) / (
                        max_of_s - min_of_s)  # 实验一下老师新说的，看会不会有效果（原来为1）
#____________________________________________________________________________________________________plus
##
    return feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new

def run_pubmed_method(sample_method, train, val, test,feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new, gcn=True):
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    # agg1 = MeanAggregator(features, cuda=True)
    if sample_method == "adj":
        agg1 = MeanAggregator(features, adj, cuda=True)
        enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg1 = MeanAggregator(features, wave, cuda=True)
        enc1 = Encoder(features, 500, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg1 = MeanAggregator(features, wave_new, cuda=True)
        enc1 = Encoder(features, 500, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    # agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    if sample_method == "adj":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), adj, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave_new, cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                       base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(3, enc2)
    graphsage.fit(train, val, labels)

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    rec = ""
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:" + str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    # print ("Average batch time:", np.mean(times))
    # rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res.txt", "pubmed" + sample_method + str(gcn) + rec)
    return f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
def run_pubmed(sample_method, train, val, test, gcn=True):
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_pubmed(10)
    method=['wave_new','wave','adj']
    test_acc_wavenew=0.0
    test_acc_wave=0.0
    test_acc_adj=0.0

    test_acc_wavenew=run_pubmed_method('wave_new', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                      wave_new, gcn=gcn)
    print('wave_new',gcn,test_acc_wavenew)
    test_acc_wave=run_pubmed_method('wave', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                      wave_new, gcn=gcn)
    print('wave', gcn, test_acc_wave)
    test_acc_adj=run_pubmed_method('adj', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                      wave_new, gcn=gcn)
    print('adj', gcn, test_acc_adj)
    return test_acc_wavenew, test_acc_wave, test_acc_adj


def load_citeseer(num_sample = 5):
    #hardcoded for simplicity...
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set) # wavelet 基过滤后的列表
    with open("../citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            if info[0] not in node_map or info[1] not in node_map:
                print(info[0], info[1])
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    
    wave_file = "citeseer"+"chev_"+str(chev_order)+"sample_"+str(num_sample)+".pkl"
    if not os.path.exists(wave_file):
        with open(wave_file, "rb") as wf:
            wave_lists = pkl.load(wf)
    else:
        adj_mat = sparse.lil_matrix((num_nodes,num_nodes))
        for p1 in adj_lists:
            for p2 in adj_lists[p1]:
                adj_mat[p1, p2] = 1
        
        G = graphs.Graph(adj_mat)
        G.estimate_lmax()
        f = filters.Heat(G, tau) # 此处的参数可变
        chebyshev = filters.approximations.compute_cheby_coeff(f, m=chev_order)
        s = filters.approximations.cheby_op(G, chebyshev, np.eye(num_nodes))
        for i in range(s.shape[0]):
            ls = []
            neis = []
            for j in range(s.shape[1]):
                if s[i][j]>thre:
                    ls.append((j,s[i][j]))
            if len(ls)<num_sample:
                for k in range(len(ls)):
                    neis.append(ls[k][0])
            else:
                ls = sorted(ls, key=lambda x:x[1], reverse=True)
                for k in range(num_sample):
                    neis.append(ls[k][0])
            wave_lists[i] = set(neis)
        with open(wave_file, "wb") as wf:
            pkl.dump(wave_lists, wf)
##
##_________________________________________________________________________________________________________________plus
    adj = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave = np.zeros((num_nodes, num_nodes), dtype='float32')
    wave_new = np.zeros((num_nodes, num_nodes), dtype='float32')
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
    #_____________________________________________________________________________________________plus
    ##
    return feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new
def run_citeseer_method(sample_method, train, val, test,feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new, num_nodes,  gcn=True):
    features = nn.Embedding(num_nodes, 3703)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    # agg1 = MeanAggregator(features,wave_new, cuda=True)
    if sample_method == "adj":
        agg1 = MeanAggregator(features, adj, cuda=True)
        enc1 = Encoder(features, 3703, 128, adj_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg1 = MeanAggregator(features, wave, cuda=True)
        enc1 = Encoder(features, 3703, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg1 = MeanAggregator(features, wave_new, cuda=True)
        enc1 = Encoder(features, 3703, 128, wave_lists, agg1, gcn=gcn, cuda=False)
    # agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), wave_new,cuda=False)
    enc1.num_sample = 300
    if sample_method == "adj":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), adj, cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave, cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)
    elif sample_method == "wave_new":
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), wave_new, cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, wave_lists, agg2,
                base_model=enc1, gcn=gcn, cuda=False)

    enc2.num_sample = 300

    graphsage = SupervisedGraphSage(6, enc2)

    graphsage.fit(train, val, labels)


    val_output = graphsage.forward(val)
    test_output=graphsage.forward(test)
    rec = ""
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nValidation F1:"+str(f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    rec += "\nTest F1:" + str(f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))
    # print ("Average batch time:", np.mean(times))
    # rec += "\nAverage batch time:"+ str(np.mean(times))
    record("res1.txt", "\nciteseer"+sample_method+str(gcn)+rec)
    return f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
def run_citeseer(sample_method, train, val, test, gcn=True):
    # np.random.seed(1)
    # random.seed(1)
    num_nodes = 3312
    feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_citeseer(3)
    #___________________________________largest graph
    import networkx as nx
    G = nx.Graph(adj)
    largest_graph = list(max(nx.connected_components(G), key=len))
    count_of_train = 0
    count_of_val = 0
    count_of_test = 0
    for i in train:
        if i in largest_graph:
            count_of_train += 1
    for i in val:
        if i in largest_graph:
            count_of_val += 1
    for i in test:
        if i in largest_graph:
            count_of_test += 1

    print("count_of_train", count_of_train/len(train))
    print("count_of_val", count_of_val / len(val))
    print("count_of_test", count_of_test / len(test))
    # adj = adj[largest_graph][:, largest_graph]
    # wave = wave[largest_graph][:, largest_graph]
    # wave_new = wave_new[largest_graph][:, largest_graph]
    # feat_data = feat_data[largest_graph]
    # labels = labels[largest_graph]
    # num_nodes = len(largest_graph)
    adj_lists = defaultdict(set)
    wave_lists = defaultdict(set)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] != 0:
                adj_lists[i].add(j)
            if wave[i][j] != 0:
                wave_lists[i].add(j)
    # ___________________________________largest graph
    method = ['wave_new', 'wave', 'adj']
    test_acc_wavenew = 0.0
    test_acc_wave = 0.0
    test_acc_adj = 0.0

    test_acc_wavenew = run_citeseer_method('wave_new', train, val, test, feat_data, labels, adj_lists, wave_lists, adj,
                                         wave,
                                         wave_new, num_nodes,  gcn=gcn)
    print('wave_new', gcn, test_acc_wavenew)
    test_acc_wave = run_citeseer_method('wave', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                                      wave_new, num_nodes,  gcn=gcn)
    print('wave', gcn, test_acc_wave)
    test_acc_adj = run_citeseer_method('adj', train, val, test, feat_data, labels, adj_lists, wave_lists, adj, wave,
                                     wave_new, num_nodes,  gcn=gcn)
    print('adj', gcn, test_acc_adj)
    return test_acc_wavenew, test_acc_wave, test_acc_adj, count_of_train, count_of_val, count_of_test


def record(filename,content):
    with open(filename, 'a') as f:
        f.writelines(content)
def run():
    num_nodes_dic = {'cora': 2708, 'citeseer': 3312, 'pubmed': 19717}
    method=["wave_new", "wave", 'adj']
    for data_set in ['cora','citeseer','pubmed' ]:  #,,'pubmed''cora'
        num_nodes = num_nodes_dic[data_set]
        for j in [False, True]:  #
            avg = [0, 0, 0]
            wave_new=[]
            wave=[]
            adj=[]
            count_tr=[]
            count_val=[]
            count_test=[]
            for k in range(20):  #
                if data_set == 'cora':


                    rand_indices = np.random.permutation(2485)
                    test = rand_indices[640:1640]  # range(1708,2708)
                    val = rand_indices[140:640]  # range(140,640)
                    train = list(rand_indices[:140])  # list(range(140))
                    # for i in range(len(method)):
                    tau = []
                    tau.append(5)
                    # print(j, method[i], k, '\n')
                    test_acc_wavenew, test_acc_wave, test_acc_adj = run_cora('test', train, val, test, j)
                    avg[0] += test_acc_wavenew
                    avg[1] += test_acc_wave
                    avg[2] += test_acc_adj
                elif data_set == 'pubmed':

                    rand_indices = np.random.permutation(num_nodes)
                    test = rand_indices[18717:]  # range(1708,2708)
                    val = rand_indices[60:560]  # range(140,640)
                    train = list(rand_indices[:60])  # list(range(140))
                    # for i in range(len(method)):
                    tau = []
                    tau.append(5)
                    # print(j, method[i], k, '\n')
                    test_acc_wavenew, test_acc_wave, test_acc_adj = run_pubmed('test', train, val, test, j)
                    avg[0]+=test_acc_wavenew
                    avg[1]+=test_acc_wave
                    avg[2]+=test_acc_adj

                elif data_set == 'citeseer':
                    rand_indices = np.random.permutation(num_nodes)
                    test = rand_indices[620:1620]  # range(1708,2708)rand_indices[:num_nodes//3]#
                    val = rand_indices[120:620]  # range(140,640)rand_indices[num_nodes//3:(2*num_nodes)//3]#
                    train = list(rand_indices[:120])  # list(range(140))list(rand_indices[(2*num_nodes)//3:])#
                    tau = []
                    tau.append(1.2)
                    # print(j, method[i], k, '\n')
                    test_acc_wavenew, test_acc_wave, test_acc_adj, count_of_train, count_of_val, count_of_test= run_citeseer('test', train, val, test, j)
                    avg[0] += test_acc_wavenew
                    avg[1] += test_acc_wave
                    avg[2] += test_acc_adj
                for i in range(len(method)):
                    wave_new.append(test_acc_wavenew)
                    wave.append(test_acc_wave)
                    adj.append(test_acc_adj)
                    count_tr.append(count_of_train/len(train))
                    count_val.append(count_of_val/len(val))
                    count_test.append(count_of_test/len(test))

                    record("largesttestavg_" + data_set + ".txt", "\n" + data_set + "wave_new "+ str(j) + ":" + str(test_acc_wavenew))
                    record("largesttestavg_" + data_set + ".txt", "\n" + data_set + "wave " + str(j) + ":" + str(test_acc_wave))
                    record("largesttestavg_" + data_set + ".txt", "\n" + data_set + "adj " + str(j) + ":" + str(test_acc_adj))
            print("DCNN_WaveShrink", wave_new)
            print('DCNN_WaveThresh', wave)
            print('DCNN', adj)
            print("The proportion of nodes in the largest subgraph in the training set", count_tr)
            print('The proportion of nodes in the largest subgraph in the val set', count_val)
            print('The proportion of nodes in the largest subgraph in the test set', count_test)
            for i in range(len(method)):
                avg_of_method=avg[i]
                avg_of_method /= 20
                record("avg_" + data_set + ".txt", "\n" + data_set + method[i] + str(j) + ":" + str(avg_of_method))
            record("avg_" + data_set + ".txt", "\n")
def run_testtau():
    num_nodes_dic = {'cora': 2708, 'citeseer': 3312, 'pubmed': 19717}
    method = ["wave_new", "wave", 'adj']
    for data_set in ['citeseer', 'cora']:  #, 'pubmed'
        num_nodes = num_nodes_dic[data_set]
        for j in [False, True]:  #
            avg = [[0 for i in range(20)], [0 for i in range(20)], [0 for i in range(20)]]
            for k in range(1):  #
                if data_set == 'cora':

                    rand_indices = np.random.permutation(num_nodes)
                    test = rand_indices[1708:]  # range(1708,2708)
                    val = rand_indices[140:640]  # range(140,640)
                    train = list(rand_indices[:140])  # list(range(140))
                    # for i in range(len(method)):
                    for tau_value in range(20):
                        tau = []
                        tau.append(3 + 0.2 * tau_value)
                        # print(j, method[i], k, '\n')
                        test_acc_wavenew, test_acc_wave, test_acc_adj = run_cora('test', train, val, test, j)
                        avg[0][tau_value] += test_acc_wavenew
                        avg[1][tau_value] += test_acc_wave
                        avg[2][tau_value] += test_acc_adj
                elif data_set == 'pubmed':

                    rand_indices = np.random.permutation(num_nodes)
                    test = rand_indices[18717:]  # range(1708,2708)
                    val = rand_indices[60:560]  # range(140,640)
                    train = list(rand_indices[:60])  # list(range(140))
                    # for i in range(len(method)):
                    for tau_value in range(20):
                        tau = []
                        tau.append(3 + 0.2 * tau_value)
                        # print(j, method[i], k, '\n')
                        test_acc_wavenew, test_acc_wave, test_acc_adj = run_pubmed('test', train, val, test, j)
                        avg[0][tau_value] += test_acc_wavenew
                        avg[1][tau_value] += test_acc_wave
                        avg[2][tau_value] += test_acc_adj

                elif data_set == 'citeseer':
                    rand_indices = np.random.permutation(num_nodes)
                    test = rand_indices[2312:]  # range(1708,2708)
                    val = rand_indices[120:620]  # range(140,640)
                    train = list(rand_indices[:120])  # list(range(140))
                    for tau_value in range(20):
                        tau = []
                        tau.append(3 + 0.2 * tau_value)
                        # print(j, method[i], k, '\n')
                        test_acc_wavenew, test_acc_wave, test_acc_adj = run_citeseer('test', train, val, test, j)
                        avg[0][tau_value] += test_acc_wavenew
                        avg[1][tau_value] += test_acc_wave
                        avg[2][tau_value] += test_acc_adj

            for i in range(len(method)):
                for tau_value in range(20):
                    avg_of_method = avg[i][tau_value]
                    avg_of_method /= 5
                    record("avg_" + data_set + ".txt",
                           "\n" + data_set + method[i] + str(j) + 'tau:(' + str(3 + 0.2 * tau_value) + ')' + ":" + str(
                               avg_of_method))
                record("avg_" + data_set + ".txt", "\n")

def run_tau():
    num_nodes_dic = {'cora': 2708, 'citeseer': 3312, 'pubmed': 19717}
    for data_set in ['pubmed', 'cora']:  # ,, ,,'citeseer',
        # np.random.seed(1)
        # random.seed(1)
        num_nodes = num_nodes_dic[data_set]
        for j in [False, True]:  #
            for i in ["wave_new", "wave", 'adj']:  #
                if data_set == 'cora':
                    avg = 0.0
                    for k in range(10):
                        tau = []
                        tau.append(5)
                        print(j, i, k, '\n')
                        rand_indices = np.random.permutation(num_nodes)
                        test = rand_indices[1708:]  # range(1708,2708)
                        val = rand_indices[140:640]  # range(140,640)
                        train = list(rand_indices[:140])  # list(range(140))
                        test_acc = run_cora(i, train, val, test, gcn=j)  # 改
                        avg += test_acc
                        print("avg-----------", test_acc)
                        # record("tau_avg_cora.txt", "\ncora" + i + str(j) + str(tau[0])+ ":" + str(avg))#改
                    avg /= 10
                    record("avg_cora.txt", "\ncora" + i + str(j) + ":" + str(avg))
                    # ___________________________
                    # avg = 0.0
                    # for k in range(10):
                    #     tau = []
                    #     tau.append(k)
                    #     print(j, i, k, '\n')
                    #     test_acc = run_cora(i, train, val, test, gcn=j)  # 改
                    #     avg += test_acc
                    #     print("avg-----------", test_acc)
                    #     # record("tau_avg_cora.txt", "\ncora" + i + str(j) + str(tau[0])+ ":" + str(avg))#改
                    # avg /= 10
                    # record("avg_cora.txt", "\ncora" + i + str(j) + ":" + str(avg))
                    # _________________________________________________
                elif data_set == 'pubmed':
                    avg = 0.0

                    for k in range(10):
                        tau = []
                        tau.append(6.3)
                        print(j, i, k, '\n')
                        rand_indices = np.random.permutation(num_nodes)
                        test = rand_indices[18717:]  # range(1708,2708)
                        val = rand_indices[60:560]  # range(140,640)
                        train = list(rand_indices[:60])  # list(range(140))

                        avg += run_pubmed(i, train, val, test, j)
                    avg /= 10
                    record("avg_pubmed.txt", "\npubmed" + i + str(j) + ":" + str(avg))
                elif data_set == 'citeseer':
                    avg = 0.0
                    for k in range(10):
                        tau = []
                        tau.append(3.2)
                        print(j, i, k, '\n')
                        rand_indices = np.random.permutation(num_nodes)
                        test = rand_indices[2312:]  # range(1708,2708)
                        val = rand_indices[120:620]  # range(140,640)
                        train = list(rand_indices[:120])  # list(range(140))
                        avg += run_citeseer(i, train, val, test, gcn=j)  # 改
                        # record("tau_avg_citeseer.txt", "\nciteseer" + i + str(j) + str(tau[0]) + ":" + str(avg))# 改
                    avg /= 10
                    record("avg_citeseer.txt", "\nciteseer" + i + str(j) + ":" + str(avg))
            record("avg_" + data_set + ".txt", "\n")

    # run_cora("adj")
    # run_cora("wave")
    # run_pubmed("adj")
    # run_pubmed("wave")
    # run_citeseer("adj")
    # run_citeseer("wave")
def draw():
    import numpy as np
    import matplotlib.pyplot as plt
    from pygsp import graphs, filters, plotting, utils

    # num_nodes = 2708
    # num_feats = 1433
    # num_sample = 10
    # feat_data = np.zeros((num_nodes, num_feats))
    # labels = np.empty((num_nodes, 1), dtype=np.int64)
    # node_map = {}
    # label_map = {}
    # with open("../cora/cora.content") as fp:
    #     for i, line in enumerate(fp):
    #         info = line.strip().split()
    #         feat_data[i, :] = list(map(float, info[1:-1]))
    #         node_map[info[0]] = i
    #         if not info[-1] in label_map:
    #             label_map[info[-1]] = len(label_map)
    #         labels[i] = label_map[info[-1]]
    #
    # adj_lists = defaultdict(set)
    # wave_lists = defaultdict(set)  # wavelet 基过滤后的列表
    # with open("../cora/cora.cites") as fp:
    #     for i, line in enumerate(fp):
    #         info = line.strip().split()
    #         paper1 = node_map[info[0]]
    #         paper2 = node_map[info[1]]
    #         adj_lists[paper1].add(paper2)
    #         adj_lists[paper2].add(paper1)
    # wave_file = "cora" + "chev_" + str(chev_order) + "sample_" + str(num_sample) + ".pkl"
    # if 0:
    #     with open(wave_file, "rb") as wf:
    #         wave_lists = pkl.load(wf)
    # else:
    #     adj_mat = sparse.lil_matrix((num_nodes, num_nodes))
    #     for p1 in adj_lists:
    #         for p2 in adj_lists[p1]:
    #             adj_mat[p1, p2] = 1
    #
    #     G = graphs.Graph(adj_mat)
    # G.set_coordinates('random3D')
    G = graphs.Bunny()

    taus = [10, 10]
    g = filters.Heat(G, taus)
    s = np.zeros(G.N)
    DELTA = 20
    s[DELTA] = 50
    s = g.filter(s, method='chebyshev')
    fig = plt.figure(figsize=(10,
                              9))
    for i in range(g.Nf):
        ax = fig.add_subplot(1, 1, i + 1, projection='3d')
        G.plot_signal(s[:, i], colorbar=True, ax=ax)
        title = r'Heat diffusion,$\tau={}$'.format(taus[i])
        _ = ax.set_title(title)
        ax.set_axis_off()
        plt.title(title, y=1.1, fontsize=30)
        plt.show()
    fig.tight_layout()
    plt.show()
def about_data_():
    import networkx as nx
    # feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_cora(5)
    # feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_pubmed(10)
    feat_data, labels, adj_lists, wave_lists, adj, wave, wave_new = load_citeseer(3)
    G = nx.Graph(adj)
    # for i in range(len(adj)):
    #     for j in range(len(adj[0])):
    #         if adj[i][j]==1.0:
    #             G.add_edge(i, j)
    # print(nx.clustering(G))
    # print('average_clustering',nx.average_clustering(G))\
    test = list(max(nx.connected_components(G), key=len))
    adj_largest = adj[test][:, test]
    feat_data_largest = feat_data[test]
    labels_larget = labels[test]

    for C in list(G.subgraph(c) for c in nx.connected_components(G)):
        print(C)
        print('average_shortest_path_length', nx.average_shortest_path_length(C))
    print('transitivity', nx.transitivity(G))
    import networkx.algorithms.community as nx_comm
    import community.community_louvain

    partition = community.community_louvain.best_partition(G)
    print('community_louvain.best_partition', partition)
if __name__ == "__main__":
    seed_torch(4)
    run()
    # about_data_()
    # run_testtau()










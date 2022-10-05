# -*- coding:UTF-8 -*-
from __future__ import division
# 即使在python2.X，使用print就得像python3.X那样加括号使用。
from __future__ import print_function
# 导入python未来支持的语言特征division(精确除法)，
# 当我们没有在程序中导入该特征时，"/"操作符执行的是截断除法(Truncating Division)；
# 当我们导入精确除法之后，"/"执行的是精确除法, "//"执行截断除除法

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pprint

import numpy
import tensorflow.compat.v1 as tf
import warnings
from utils import *
from models import GCN, MLP, Wave_Cheby_Neural_Network, Spectral_CNN, GCN2

import os

from utils2 import load_data2
import pandas as pd
import time

tf.disable_v2_behavior()
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# True: using pretrained model for cora in dir:pre_trained
# False: training a new model
Pretrain_model_cora = False
# save model
# checkpt_new_file = 'pre_trained/model_cora_new.ckpt'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'wave_gcn_neural_network', 'Model string.')
# 'gcn', 'gcn_cheby', 'dense','spectral_basis'
# 'wave_cheby_neural_network','wave_gcn_neural_network'
flags.DEFINE_float('wavelet_s', 1.2, 'wavelet s .')
flags.DEFINE_float('threshold', 1e-5, 'sparseness threshold .')
flags.DEFINE_string('wave_normalize_type', 'original', 'wavelet weight normalize type.')  # 'original', 'norm1'
flags.DEFINE_bool('weight_share', False, 'Weight share string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('alldata', False, 'All data string.')
flags.DEFINE_bool('random_train', True, 'Training is random.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')  # 1000

# K阶的切比雪夫近似矩阵的参数k
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')  #
flags.DEFINE_bool('mask', False, 'mask string.')
flags.DEFINE_bool('normalize', False, 'normalize string.')
flags.DEFINE_bool('laplacian_normalize', True, 'laplacian normalize string.')
flags.DEFINE_bool('sparse_ness', True, 'wavelet sparse_ness string.')
flags.DEFINE_integer('order', 2, 'neighborhood order .')
flags.DEFINE_bool('weight_normalize', False, 'weight normalize string.')

# 第一层的输出维度；卷积层第一层的output_dim，第二层的input_dim
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')

# 避免过拟合（按照一定的概率随机丢弃一部分神经元）
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')

# 权值衰减：防止过拟合
# loss计算方式（权值衰减+正则化）：self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')  # 200

wavelet_s_list = [2, 2.2, 2.4, 2.6, 2.8, 3, 3.2]  # s参数列表
res = []
for s in wavelet_s_list:
    res.append([s])


def train_once_random_data():
    # Load data
    # 内容格式（后来的）
    labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset,
                                                                                               alldata=FLAGS.alldata,
                                                                                               random_train=FLAGS.random_train,
                                                                                               max_subgraph=False)

    features = preprocess_features(features)

    def train_once_wavelet_s(wavelet_s_index):
        def print_flags(FLAGS):
            print("FLAGS.wavelet_s : ", FLAGS.wavelet_s)
            print("FLAGS.model : ", FLAGS.model)
            print("FLAGS.threshold : ", FLAGS.threshold)

        print("************Loading data finished, Begin constructing wavelet************")
        print_flags(FLAGS)

        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN

        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN

        elif FLAGS.model == "spectral_basis":
            dataset = FLAGS.dataset
            wavelet_s = FLAGS.wavelet_s
            laplacian_normalize = FLAGS.laplacian_normalize
            sparse_ness = FLAGS.sparse_ness
            threshold = FLAGS.threshold
            weight_normalize = FLAGS.weight_normalize
            wave_normalize_type = FLAGS.wave_normalize_type
            support = wavelet_basis(dataset, adj, wavelet_s, laplacian_normalize, sparse_ness, threshold,
                                    weight_normalize, wave_normalize_type)  # wavelet_basis  spectral_basis
            num_supports = len(support)
            model_func = Spectral_CNN

        elif FLAGS.model == "wave_cheby_neural_network":
            dataset = FLAGS.dataset
            wavelet_s = FLAGS.wavelet_s
            laplacian_normalize = FLAGS.laplacian_normalize
            sparse_ness = FLAGS.sparse_ness
            threshold = FLAGS.threshold
            weight_normalize = FLAGS.weight_normalize
            wave_normalize_type = FLAGS.wave_normalize_type
            support = wave_cheby_basis(dataset, adj, wavelet_s, laplacian_normalize, sparse_ness, threshold,
                                       weight_normalize, wave_normalize_type)  #
            support.extend(chebyshev_polynomials(adj, FLAGS.max_degree))
            num_supports = len(support)
            model_func = Wave_Cheby_Neural_Network


        elif FLAGS.model == "wave_gcn_neural_network":
            dataset = FLAGS.dataset
            wavelet_s = FLAGS.wavelet_s
            laplacian_normalize = FLAGS.laplacian_normalize
            sparse_ness = FLAGS.sparse_ness
            threshold = FLAGS.threshold
            weight_normalize = FLAGS.weight_normalize
            wave_normalize_type = FLAGS.wave_normalize_type
            support = wave_cheby_basis(dataset, adj, wavelet_s, laplacian_normalize, sparse_ness, threshold,
                                       weight_normalize, wave_normalize_type)  #
            support.extend([preprocess_adj(adj)])
            num_supports = len(support)
            model_func = Wave_Cheby_Neural_Network

        elif FLAGS.model == "gcn2":
            dataset = FLAGS.dataset
            wavelet_s = FLAGS.wavelet_s
            laplacian_normalize = FLAGS.laplacian_normalize
            sparse_ness = FLAGS.sparse_ness
            threshold = FLAGS.threshold
            weight_normalize = FLAGS.weight_normalize
            support = wave_cheby_basis(dataset, adj, wavelet_s, laplacian_normalize, sparse_ness, threshold,
                                       weight_normalize)  #
            support.extend([preprocess_adj(adj)])
            num_supports = len(support)
            model_func = GCN2

        elif FLAGS.model == "gcn_cheby2":
            dataset = FLAGS.dataset
            wavelet_s = FLAGS.wavelet_s
            laplacian_normalize = FLAGS.laplacian_normalize
            sparse_ness = FLAGS.sparse_ness
            threshold = FLAGS.threshold
            weight_normalize = FLAGS.weight_normalize
            support = wave_cheby_basis(dataset, adj, wavelet_s, laplacian_normalize, sparse_ness, threshold,
                                       weight_normalize)  #
            support.extend(chebyshev_polynomials(adj, FLAGS.max_degree))
            num_supports = len(support)
            model_func = GCN2

        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            num_supports = 1
            model_func = MLP
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            # 由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            # features也是稀疏矩阵，也用LIL格式表示，因此定义为tf.sparse_placeholder(tf.float32)，维度(2708, 1433)
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        weight_normalize = FLAGS.weight_normalize
        node_num = adj.shape[0]
        model = model_func(node_num, weight_normalize, placeholders, input_dim=features[2][1], logging=True)

        print("**************Constructing wavelet finished, Begin training**************")
        # Initialize session
        # Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分
        sess = tf.Session()

        # Define model evaluation function
        def evaluate(features, support, labels, mask, placeholders):
            feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
            outs_val = sess.run([model.outputs, model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], outs_val[2]

        saver = tf.train.Saver()

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Train model
        cost_val = []
        best_val_acc = 0.0
        output_test_acc = 0.0

        if not Pretrain_model_cora:
            for epoch in range(FLAGS.epochs):
                # print("support:")
                # print(support[1])
                # Construct feed dictionary
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

                # Validation
                val_output, cost, acc = evaluate(features, support, y_val, val_mask, placeholders)
                cost_val.append(cost)

                # if(best_val_acc<=acc):
                #     saver.save(sess, checkpt_new_file)

                # Test
                test_output, test_cost, test_acc = evaluate(features, support, y_test, test_mask, placeholders)

                # 记录acc 最大的时候
                if best_val_acc <= acc:
                    best_val_acc = acc
                    output_test_acc = test_acc

                # Print results
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                      "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                      "val_acc=", "{:.5f}".format(acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=",
                      "{:.5f}".format(test_acc))

                if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                    print("Early stopping...")
                    break


        if not Pretrain_model_cora:
            print("Optimization Finished!")

            print("dataset: ", FLAGS.dataset, " model: ", FLAGS.model, "order: ", FLAGS.order, ",sparse_ness: ",
                  FLAGS.sparse_ness,
                  ",laplacian_normalize: ", FLAGS.laplacian_normalize, ",threshold", FLAGS.threshold, ",wavelet_s:",
                  FLAGS.wavelet_s, ",mask:", FLAGS.mask,
                  ",normalize:", FLAGS.normalize, ",weight_normalize:", FLAGS.weight_normalize, " weight_share:",
                  FLAGS.weight_share,
                  ",learning_rate:", FLAGS.learning_rate, ",hidden1:", FLAGS.hidden1, ",dropout:", FLAGS.dropout,
                  ",max_degree:", FLAGS.max_degree, ",alldata:", FLAGS.alldata)

            print("Val accuracy:", best_val_acc, " Test accuracy: ", output_test_acc)
            res[wavelet_s_index].append(output_test_acc)

        print("********************************************************")

    s_len = len(wavelet_s_list)
    if s_len > 0:
        for s_i, the_s in enumerate(wavelet_s_list):
            FLAGS.wavelet_s = the_s
            train_once_wavelet_s(s_i)
    else:
        res.append([FLAGS.wavelet_s])
        train_once_wavelet_s(0)


# 训练次数
train_times = 5
for i in range(train_times):
    train_once_random_data()
# 记录训练结果到excel中
dataFrame1 = pd.DataFrame(res)
with pd.ExcelWriter(
        FLAGS.model + '_' + FLAGS.dataset + '_train_result' + time.strftime("_%Y%m%d%H%M%S") + '.xlsx') as writer:
    dataFrame1.to_excel(writer)

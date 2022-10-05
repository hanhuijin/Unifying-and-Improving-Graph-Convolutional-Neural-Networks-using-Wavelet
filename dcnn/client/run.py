import numpy as np
from sklearn import metrics
import random

from client import parser

from python import data
from python import models
from python import params as params_mod


metadata = {
    "cora": {
        "parser": data.parse_cora,
        "num_nodes": 2485,
        "num_features": 1433,
        "num_classes": 7,
    },
    "pubmed": {
        "parser": data.parse_pubmed,
        "num_nodes": 19717,
        "num_features": 500,
        "num_classes": 3,
    },
    "citeseer": {
        "parser": data.parse_citeseer,
        "num_nodes": 2110,
        "num_features": 3703,
        "num_classes": 6,
    },
    "sparse_cora": {
        "parser": data.parse_cora_sparse,
        "num_nodes": 2708,
        "num_features": 1433,
        "num_classes": 7,
    },
    'nci1': {
        "parser": lambda: data.parse_graph_data('nci1.graph'),
        "num_nodes": None,
        "num_graphs": None,
        "num_features": 37,
        "num_classes": 2,
    }

}

model_map = {
    "node_classification": models.NodeClassificationDCNN,
    "deep_node_classification": models.DeepNodeClassificationDCNN,
    "post_sparse_node_classification": models.PostSparseNodeClassificationDCNN,
    "pre_sparse_node_classification": models.PreSparseNodeClassificationDCNN,
    "true_sparse_node_classification": models.TrueSparseNodeClassificationDCNN,
    "deep_dense_node_classification": models.DeepDenseNodeClassificationDCNN,
    "graph_classification": models.GraphClassificationDCNN,
    "deep_graph_classification": models.DeepGraphClassificationDCNN,
    "deep_graph_classification_with_reduction": models.DeepGraphClassificationDCNNWithReduction,
    "deep_graph_classification_with_kron_reduction": models.DeepGraphClassificationDCNNWithKronReduction,
    "feature_aggregated_graph_classification": models.GraphClassificationFeatureAggregatedDCNN,
}

hyperparameter_choices = {
    "num_hops": range(1, 6),
    "learning_rate": [0.01, 0.05, 0.1, 0.25],
    "optimizer": params_mod.update_map.keys(),
    "loss": params_mod.loss_map.keys(),
    "dcnn_nonlinearity": params_mod.nonlinearity_map.keys(),
    "dense_nonlinearity": params_mod.nonlinearity_map.keys(),
    "num_epochs": [10, 20, 50, 100],
    "batch_size": [10, 100],
    "early_stopping": [0, 1],
    "stop_window_size": [1, 5, 10],
    "num_dcnn_layers": range(2, 6),
    "num_dense_layers": range(2, 6),
}

def run_node_classification(parameters,tau,train_indices,valid_indices,test_indices):
    A1,A2,A3, X, Y = metadata[parameters.data]["parser"](tau)


    seed_torch(4)
    dcnn1 = model_map[parameters.model](parameters, A1)
    seed_torch(4)
    dcnn2 = model_map[parameters.model](parameters, A2)
    seed_torch(4)
    dcnn3 = model_map[parameters.model](parameters, A3)
    # num_nodes = A1.shape[0]

    # indices = np.arange(num_nodes).astype('int32')
    # np.random.shuffle(indices)
    #
    # train_indices = indices[:num_nodes // 3]
    # valid_indices = indices[num_nodes // 3: (2 * num_nodes) // 3]
    # test_indices = indices[(2 * num_nodes) // 3:num_nodes]

    import datetime
    actuals = Y[test_indices, :].argmax(1)

    start = datetime.datetime.now()
    train_loss_list, train_acc_list, val_loss_list, val_acc_list= dcnn3.fit(X, Y, train_indices, valid_indices)
    np.save('wave_newtrain_loss_list.npy',train_loss_list)
    np.save('wave_newtrain_acc_list.npy', train_acc_list)
    np.save('wave_newval_loss_list.npy', val_loss_list)
    np.save('wave_newval_acc_list.npy', val_acc_list)
    end = datetime.datetime.now()
    print('time of wave_new', end - start)
    predictions3 = dcnn3.predict(X, test_indices)
    accuracy3 = metrics.accuracy_score(actuals, predictions3)
    print("Test Accuracy3(wave_new): %.4f" % (accuracy3,))


    start = datetime.datetime.now()
    train_loss_list, train_acc_list, val_loss_list, val_acc_list= dcnn1.fit(X, Y, train_indices, valid_indices)
    np.save('adjtrain_loss_list.npy',train_loss_list)
    np.save('adjtrain_acc_list.npy', train_acc_list)
    np.save('adjval_loss_list.npy', val_loss_list)
    np.save('adjval_acc_list.npy', val_acc_list)
    end = datetime.datetime.now()
    print('time of adj',end - start)
    predictions1 = dcnn1.predict(X, test_indices)
    accuracy1 = metrics.accuracy_score(actuals, predictions1)
    print("Test Accuracy1(adj): %.4f" % (accuracy1,))



    start = datetime.datetime.now()
    train_loss_list, train_acc_list, val_loss_list, val_acc_list= dcnn2.fit(X, Y, train_indices, valid_indices)
    np.save('wavetrain_loss_list.npy', train_loss_list)
    np.save('wavetrain_acc_list.npy', train_acc_list)
    np.save('waveval_loss_list.npy', val_loss_list)
    np.save('waveval_acc_list.npy', val_acc_list)
    end = datetime.datetime.now()
    print('time of wave',end - start)
    predictions2 = dcnn2.predict(X, test_indices)
    accuracy2 = metrics.accuracy_score(actuals, predictions2)
    print("Test Accuracy2(wave): %.4f" % (accuracy2,))
    return accuracy3,accuracy1,accuracy2


def run_graph_classification(params):
    print ("parsing data...")
    A ,X, Y = metadata[parameters.data]["parser"]()

    # Shuffle the data.
    tmp = list(zip(A, X, Y))
    random.shuffle(tmp)
    A, X, Y = zip(*tmp)

    num_graphs = len(A)

    indices = np.arange(num_graphs).astype('int32')
    np.random.shuffle(indices)

    train_indices = indices[:num_graphs // 3]
    valid_indices = indices[num_graphs // 3: (2 * num_graphs) // 3]
    test_indices = indices[(2 * num_graphs) // 3:]

    print ("initializing model...")
    m = model_map[params.model](params)

    print ("training model...")
    m.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

    test_predictions = []
    test_actuals = []
    for test_index in test_indices:
        pred = m.predict(A[test_index], X[test_index])
        test_predictions.append(pred)
        test_actuals.append(Y[test_index].argmax())

    test_accuracy = 0.0
    num_test = len(test_predictions)
    for i in range(len(test_predictions)):
        if (test_predictions[i] == test_actuals[i]):
            test_accuracy += 1.0

    test_accuracy /= num_test
    print ("Test Accuracy: %.6f" % (test_accuracy,))
    print (params)
    print ("RESULTS:%.6f" % test_accuracy)
    print ("done")
def draw():
    wave_newval_acc_list = np.load("wave_newval_acc_list.npy")
    wave_newval_acc_list = wave_newval_acc_list.tolist()
    adjval_acc_list = np.load("adjval_acc_list.npy")
    adjval_acc_list = adjval_acc_list.tolist()
    waveval_acc_list = np.load("waveval_acc_list.npy")
    waveval_acc_list = waveval_acc_list.tolist()
    wave_newval_loss_list = np.load("wave_newval_loss_list.npy")
    wave_newval_loss_list = wave_newval_loss_list.tolist()
    adjval_loss_list = np.load("adjval_loss_list.npy")
    adjval_loss_list = adjval_loss_list.tolist()
    waveval_loss_list = np.load("waveval_loss_list.npy")
    waveval_loss_list = waveval_loss_list.tolist()
    epochs = range(1, len(wave_newval_acc_list) + 1)
    import matplotlib.pyplot as plt

    # plt.plot(epochs, wave_newval_loss_list, 'b', color='C0', label='2-hop DCNN_WaveShrink')
    # plt.plot(epochs, waveval_loss_list, 'b', color='C1', label='2-hop DCNN_WaveThresh')
    # plt.plot(epochs, adjval_loss_list, 'b', color='C2', label='2-hop DCNN')
    # plt.title('validation loss of our model(During training)')
    # plt.xlabel('Epochs')
    # plt.ylabel('loss')
    # plt.legend()

    plt.show()
    plt.plot(epochs, wave_newval_acc_list, 'b', color='C0', label='2-hop DCNN_WaveShrink')
    plt.plot(epochs, waveval_acc_list, 'b', color='C1',  label='2-hop DCNN_WaveThresh')
    plt.plot(epochs, adjval_acc_list, 'b', color='C2', label='2-hop DCNN')
    plt.title('validation acc of our model(During training)')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()
if __name__ == '__main__':
    # draw()

    import os
    def seed_torch(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)



    seed_torch(4)
    parameters = parser.parse()

    if parameters.explore:
        while True:
            for possibility in hyperparameter_choices.keys():
                choice = random.choice(hyperparameter_choices[possibility])
                parameters.set(possibility, choice)

            print ("Parameter choices:")
            print (parameters)

            try:
                if "node_classification" in parameters.model:
                    num_nodes_dic = {'cora': 2708, 'citeseer': 3312, 'pubmed': 19717}
                    for data_set in [ 'cora']:  # ,'pubmed', ,'cora'
                        np.random.seed(1)
                        random.seed(1)
                        num_nodes = num_nodes_dic[data_set]
                        indices = np.random.permutation(num_nodes)
                        train_indices = indices[:num_nodes // 3]
                        valid_indices = indices[num_nodes // 3: (2 * num_nodes) // 3]
                        test_indices = indices[(2 * num_nodes) // 3:num_nodes]

                        for i in range(6):
                            tau=[]
                            tau.append(3+i*0.5)
                            run_node_classification(parameters, tau, train_indices, valid_indices, test_indices)
                else:
                    run_graph_classification(parameters)
            except ValueError:
                print ("Encountered nan loss, trying again.")

    if "node_classification" in parameters.model:
        import datetime


        num_nodes_dic = {'cora': 2485, 'citeseer': 2110, 'pubmed': 19717}
        for data_set in ['cora']:  # ,'pubmed', ,'citeseer'
            np.random.seed(1)
            random.seed(1)
            seed_torch(4)
            num_nodes = num_nodes_dic[data_set]
            wavenew_test = [0 for i in range(10)]
            wave_test=[0 for i in range(10)]
            adj_test = [0 for i in range(10)]

            for i in range(1):
                seed_torch(i)# change_plus
                test=list(range(10))
                test=np.random.permutation(test)
                indices = np.random.permutation(num_nodes)
                train_indices = indices[:140]
                valid_indices = indices[140: 640]
                test_indices = indices[1708:]
                # A1, A2, A3, X, Y = metadata[parameters.data]["parser"](tau)
                for tau_value in range(10):
                    tau = []
                    tau.append(0.2+(tau_value*0.1) )#
                # print('tau__________________________________-',tau[0])
                    start = datetime.datetime.now()
                    # seed_torch(4)
                    accuracy3, accuracy1, accuracy2  = run_node_classification(parameters, tau, train_indices, valid_indices, test_indices)
                    wave_test[tau_value] += accuracy2
                    wavenew_test[tau_value] += accuracy3
                    adj_test[tau_value] += accuracy1
                    end = datetime.datetime.now()
                    print(end - start)
            for tau_value in range(10):
                wavenew_test[tau_value]/=5
                wave_test[tau_value]/=5
                adj_test[tau_value]/=5
            print("DCNN_WaveThresh:", wave_test, "DCNN_WaveShrink:", wavenew_test, "DCNN:", adj_test)
        # run_node_classification(parameters)


    else:
        run_graph_classification(parameters)

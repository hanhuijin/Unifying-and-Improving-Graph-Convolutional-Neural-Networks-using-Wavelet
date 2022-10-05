from inits import *
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import normalize

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


# 稀疏矩阵的dropout操作
def sparse_dropout(x, keep_prob, noise_shape):  # dropout防止overfitting
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


# 定义Layer 层，主要作用是：对每层的name做了命名，还用一个参数决定是否做log
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    # __call__ 的作用让 Layer 的实例成为可调用对象；
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


# 根据 Layer 继承得到denseNet
class Dense(Layer):
    """全连接层."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act  # 激活函数
        self.sparse_inputs = sparse_inputs  # 是否是稀疏数据
        self.featureless = featureless  # 输入的数据带不带特征矩阵
        self.bias = bias  # 是否有偏置

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    # _call 函数定义计算
    # 重写了_call 函数，其中对稀疏矩阵做 drop_out:sparse_dropout()
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


# 从 Layer 继承下来得到图卷积网络，与denseNet的唯一差别是_call函数和__init__函数（self.support = placeholders['support']的初始化）
class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        # 下面是定义变量，主要是通过调用utils.py中的glorot函数实现
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                # self.vars['weights_diffusion_'+str(i)] = tf.Variable(1.0)  GraphConvolution_WeightShar Model
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        # convolve 卷积的实现。主要是根据论文中公式Z = \tilde{D}^{-1/2}\tilde{A}^{-1/2}X\theta实现
        supports = list()  # support是邻接矩阵的一个变化
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)  # X * theta
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)  # TiL * X * theta
            # support = dot(self.support[i], self.vars['weights_diffusion_'+str(i)] * pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution_WeightShare(Layer):
    """Graph convolution layer."""

    def __init__(self, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_WeightShare, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))  #
            for i in range(len(self.support)):  #
                self.vars['weights_diffusion_' + str(i)] = tf.Variable(1.0)
                # tf.Variable：图变量的初始化方法name_variable =tf.Variable(value)  value是所有可以转换为Tensor的类型

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(0)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(0)]
            # support = dot(self.support[i], pre_sup, sparse=True)
            support = dot(self.support[i], self.vars['weights_diffusion_' + str(i)] * pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # # supports = list()
        # output = self.vars['weights_diffusion_'+str(0)] *  self.support[0]
        # output = None
        # for i in range( len( self.support)):
        #     # print(self.support[i])
        #     # support = self.vars['weights_diffusion_'+str(i)] * self.support[i]
        #     support = self.support[i]
        #     # supports.append(support)
        #     if(i == 0):
        #         output = support
        #     else:
        #         output = tf.sparse_add(output,support)
        # #
        # # # output = tf.add_n(supports)
        # # output = supports
        # # output normalize
        # print type(output)
        # if(self.weight_normalize):
        #     output = normalize(output,norm='l1',axis=1)
        #
        # pre_sup = dot(x, self.vars['weights_' + str(0)],
        #               sparse=self.sparse_inputs)
        #
        # output = dot(output,pre_sup,sparse=True)
        # print type(output)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Wavelet_Convolution(Layer):
    """Graph convolution layer."""

    def __init__(self, node_num, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Wavelet_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num  #
        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))  #
            # diag filter kernel
            self.vars['kernel'] = ones([self.node_num], name='kernel')  #

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), tf.diag(self.vars['kernel']), a_is_sparse=True,
                             b_is_sparse=True)  # tf.matmul：将矩阵 a 乘以矩阵 b,生成a * b  ## psai*F
        supports = tf.matmul(supports, tf.sparse_tensor_to_dense(self.support[1]), a_is_sparse=True,
                             b_is_sparse=True)  # ## psai*F * psai-1
        pre_sup = dot(x, self.vars['weights_' + str(0)], sparse=self.sparse_inputs)  # ## theta * X
        output = dot(supports, pre_sup)  # ## psai*F * psai-1 * theta * X

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Wave_Cheby_Convolution(Layer):
    """Graph convolution layer."""

    def __init__(self, node_num, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Wave_Cheby_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num  #
        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(3, len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        # supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), tf.diag(self.vars['kernel']), a_is_sparse=True,
        #                      b_is_sparse=True)  # tf.matmul：将矩阵 a 乘以矩阵 b,生成a * b  ## F
        supports = list()  # support是邻接矩阵的一个变化
        for i in range(3, len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)  # X * theta
            else:
                pre_sup = self.vars['weights_' + str(i)]
            # self.support[i] = tf.sparse.reorder(self.support[i])
            # support = dot(tf.sparse_tensor_to_dense(self.support[i]), tf.sparse_tensor_to_dense(self.support[0]))  # TiL * U
            # support = tf.matmul(tf.transpose(tf.sparse_tensor_to_dense(self.support[0])), support,
            #                     a_is_sparse=False, b_is_sparse=False)  # Ut * TiL * U

            # support1 = tf.sparse_tensor_to_dense(self.support[1])
            # zero1 = tf.zeros_like(support1)
            # support1 = tf.where(support1 < 1e-3, zero1, support1)
            # support2 = tf.sparse_tensor_to_dense(self.support[2])
            # zero2 = tf.zeros_like(support2)
            # support2 = tf.where(support2 < 1e-3, zero2, support2)  # thresh

            support1 = tf.sparse_tensor_to_dense(self.support[1])
            zero1 = tf.zeros_like(support1)
            support1 = tf.where(support1 < 1e-3, zero1, support1-(1e-3))
            support2 = tf.sparse_tensor_to_dense(self.support[2])
            zero2 = tf.zeros_like(support2)
            support2 = tf.where(support2 < 1e-3, zero2, support2-(1e-3))  # shrink

            support = dot(self.support[i], tf.sparse_tensor_to_dense(self.support[0]), True)
            support = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), support, transpose_a=True,
                                a_is_sparse=True, b_is_sparse=True)
            support = tf.matmul(support1, support, a_is_sparse=True, b_is_sparse=True)
            support = tf.matmul(support, support2, a_is_sparse=True, b_is_sparse=True)
            support = dot(support, pre_sup)
            # support = dot(self.support[i], self.vars['weights_diffusion_'+str(i)] * pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution2(Layer):
    """Graph convolution layer."""

    def __init__(self, node_num, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution2, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num  #
        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(3, len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        # supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), tf.diag(self.vars['kernel']), a_is_sparse=True,
        #                      b_is_sparse=True)  # tf.matmul：将矩阵 a 乘以矩阵 b,生成a * b  ## F
        supports = list()  # support是邻接矩阵的一个变化
        for i in range(3, len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)  # X * theta
            else:
                pre_sup = self.vars['weights_' + str(i)]
            # self.support[i] = tf.sparse.reorder(self.support[i])
            # support = dot(tf.sparse_tensor_to_dense(self.support[i]), tf.sparse_tensor_to_dense(self.support[0]))  # TiL * U
            # support = tf.matmul(tf.transpose(tf.sparse_tensor_to_dense(self.support[0])), support,
            #                     a_is_sparse=False, b_is_sparse=False)  # Ut * TiL * U
            support = dot(self.support[i], tf.sparse_tensor_to_dense(self.support[0]), True)  # TiL * U
            support = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), support, transpose_a=True,
                                a_is_sparse=True, b_is_sparse=True)  # Ut * TiL * U
            support = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), support,
                                a_is_sparse=True, b_is_sparse=True)  # U * Ut * TiL * U
            support = tf.matmul(support, tf.sparse_tensor_to_dense(self.support[0]), transpose_b=True,
                                a_is_sparse=True, b_is_sparse=True)  # U * Ut * TiL * U * Ut
            support = dot(support, pre_sup)  # U * Ut * TiL * U * Ut * X * theta
            # support = dot(self.support[i], self.vars['weights_diffusion_'+str(i)] * pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Spetral_Convolution(Layer):
    """Graph convolution layer."""

    def __init__(self, node_num, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Spetral_Convolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num  #
        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))  #
            # diag filter kernel
            self.vars['kernel'] = ones([self.node_num], name='kernel')  #

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  ## 迭代调用计算
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), tf.diag(self.vars['kernel']),
                             a_is_sparse=False, b_is_sparse=True)
        # supports = tf.matmul(supports,tf.sparse_tensor_to_dense(self.support[1]),a_is_sparse=False,b_is_sparse=False)
        pre_sup = dot(x, self.vars['weights_' + str(0)],
                      sparse=self.sparse_inputs)
        pre_sup = tf.matmul(tf.transpose(tf.sparse_tensor_to_dense(self.support[0])), pre_sup,
                            a_is_sparse=False, b_is_sparse=False)

        # zero = tf.zeros_like(pre_sup)
        # pre_sup = tf.where(pre_sup < 1e-3, zero, pre_sup)  # thresh版本
        # pre_sup = tf.where(tf.less(pre_sup, 1e-3), zero, pre_sup-1e-3)  # shrink版本

        output = dot(supports, pre_sup)  # U*F * Ut * theta*X

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Spetral_Convolution_try(Layer):
    """Graph convolution layer."""

    def __init__(self, node_num, weight_normalize, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(Spetral_Convolution_try, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.node_num = node_num  #
        self.weight_normalize = weight_normalize  #
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))  #
            # diag filter kernel
            self.vars['kernel'] = ones([self.node_num], name='kernel')  #

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):  ## 迭代调用计算
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = tf.sparse_tensor_to_dense(self.support[0])  
        # supports = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]), tf.diag(self.vars['kernel']),
        #                      a_is_sparse=False, b_is_sparse=True)
        pre_sup = dot(x, self.vars['weights_' + str(0)],
                      sparse=self.sparse_inputs)
        pre_sup = tf.matmul(tf.sparse_tensor_to_dense(self.support[1]), pre_sup,
                            a_is_sparse=False, b_is_sparse=False)

        # zero = tf.zeros_like(pre_sup)
        # pre_sup = tf.where(pre_sup < 1e-3, zero, pre_sup)  # thresh版本
        # pre_sup = tf.where(tf.less(pre_sup, 1e-3), zero, pre_sup-1e-3)  # shrink版本

        output = dot(supports, pre_sup)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

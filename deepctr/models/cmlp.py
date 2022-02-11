# -*- coding:utf-8 -*-
from collections import OrderedDict

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import add_func, combined_dnn_input, concat_func

from deepctr.models.pairwise import DisenPairwise

def split_features(features, uid_feature_name, disentangle_feature_name):

    user_features = OrderedDict()
    disentangle_features = OrderedDict()
    item_features = OrderedDict()
    for feature, inputs in features.items():
        if feature == uid_feature_name:
            user_features[feature] = inputs
        elif feature == disentangle_feature_name:
            disentangle_features[feature] = inputs
        else:
            item_features[feature] = inputs

    return user_features, disentangle_features, item_features


def split_feature_columns(feature_columns, uid_feature_name, disentangle_feature_name):

    user_feature_columns = list(filter(lambda x: x.name == uid_feature_name, feature_columns))
    disentangle_feature_columns = list(filter(lambda x: x.name == disentangle_feature_name, feature_columns))
    item_feature_columns = list(filter(lambda x: x.name != uid_feature_name and x.name != disentangle_feature_name, feature_columns))

    return user_feature_columns, disentangle_feature_columns, item_feature_columns


@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy
    return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
        
    def call(self, inputs):
        return GradientReversalOperator(inputs)


def CMLP(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
        task='binary', uid_feature_name=None, disentangle_feature_name=None, train_mode='point',
        disentangle_loss_weight=1.0):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    user_features, disentangle_features, item_features = split_features(features, uid_feature_name, disentangle_feature_name)
    user_dnn_feature_columns, disentangle_dnn_feature_columns, item_dnn_feature_columns = split_feature_columns(dnn_feature_columns, uid_feature_name, disentangle_feature_name)

    user_sparse_embedding_list, _ = input_from_feature_columns(user_features, user_dnn_feature_columns, l2_reg_embedding, seed)
    disentangle_sparse_embedding_list, _ = input_from_feature_columns(disentangle_features, disentangle_dnn_feature_columns, l2_reg_embedding, seed)
    item_sparse_embedding_list, _ = input_from_feature_columns(item_features, item_dnn_feature_columns, l2_reg_embedding, seed)

    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, [])
    user_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(user_dnn_input)
    user_dnn_out_disentangle = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(user_dnn_input)

    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, [])
    item_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(item_dnn_input)
    item_dnn_out_disentangle = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(item_dnn_input)

    user_item_dnn_out = user_dnn_out*item_dnn_out
    user_item_dnn_out_disentangle = user_dnn_out_disentangle*item_dnn_out_disentangle

    user_item_dnn_out_logit = tf.keras.layers.Dense(1, 
        use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed), name='logit')(user_item_dnn_out)
    user_item_dnn_out_disentangle_logit = tf.keras.layers.Dense(1, 
        use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed), name='disentangle_logit')(user_item_dnn_out_disentangle)

    dnn_logit = user_item_dnn_out_logit + user_item_dnn_out_disentangle_logit

    final_logit = add_func([dnn_logit, linear_logit])

    output = PredictionLayer(task, name='ctr')(final_logit)

    output_disentangle = tf.keras.layers.Dense(disentangle_dnn_feature_columns[0].vocabulary_size, 
        use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed), name='disentangle')(item_dnn_out_disentangle)

    dnn_out_reverse = GradientReversalLayer()(item_dnn_out)
    output_disentangle_adversarial = tf.keras.layers.Dense(disentangle_dnn_feature_columns[0].vocabulary_size, 
        use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed), name='disentangle_adversarial')(dnn_out_reverse)

    if train_mode == 'point':
        model = tf.keras.models.Model(
            inputs=inputs_list, 
            outputs=[output, output_disentangle, output_disentangle_adversarial, user_item_dnn_out_logit, user_item_dnn_out_disentangle_logit]
        )
    else:
        model = DisenPairwise(
            inputs=inputs_list, 
            outputs=[output, output_disentangle, output_disentangle_adversarial, user_item_dnn_out_logit, user_item_dnn_out_disentangle_logit],
            disentangle_loss_weight=disentangle_loss_weight
        )
    return model

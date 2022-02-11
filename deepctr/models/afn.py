# -*- coding:utf-8 -*-
from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, LNN, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input

from deepctr.models.pairwise import Pairwise


def AFN(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], lnn_dim=1500, lnn_activation='relu',
           dnn_hidden_units=(400, 400), dnn_activation='relu', l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, 
           dnn_dropout=0, dnn_use_bn=False, task='binary', model_mode='point'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    use_bias = model_mode == 'point'

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    lnn_input = concat_func(sparse_embedding_list, axis=1)

    lnn_output = LNN(lnn_dim, lnn_activation, l2_reg=l2_reg_dnn, seed=seed, use_bias=use_bias)(lnn_input)
    lnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, train_bias=use_bias)(lnn_output)
    lnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(lnn_output)

    final_logit = add_func([linear_logit, lnn_logit])

    output = PredictionLayer(task, use_bias=use_bias)(final_logit)
    if model_mode == 'index':
        model = tf.keras.models.Model(inputs=inputs_list, outputs=[output, lnn_output])
    elif model_mode == 'point':
        model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    else:
        model = Pairwise(inputs=inputs_list, outputs=output)
    return model

# -*- coding:utf-8 -*-
import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import combined_dnn_input

from deepctr.models.pairwise import Pairwise


def MLP(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
           l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary', model_mode='point'):
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

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, train_bias=use_bias)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    output = PredictionLayer(task, use_bias=use_bias)(dnn_logit)
    if model_mode == 'index':
        model = tf.keras.models.Model(inputs=inputs_list, outputs=[output, dnn_output])
    elif model_mode == 'point':
        model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    else:
        model = Pairwise(inputs=inputs_list, outputs=output)
    return model

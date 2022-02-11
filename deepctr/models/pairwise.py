# -*- coding:utf-8 -*-
from itertools import chain

import tensorflow as tf
from tensorflow import keras

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input


class Pairwise(keras.Model):
    def __init__(self, inputs, outputs):

        super(Pairwise, self).__init__(inputs=inputs, outputs=outputs)

    def train_step(self, data):
        [pos_x, neg_x], [pos_y, neg_y] = data


        with tf.GradientTape() as tape:
            pos_y_pred = self(pos_x, training=True)
            neg_y_pred = self(neg_x, training=True)

            ctr_loss = tf.math.reduce_sum(tf.math.maximum(0.0, neg_y_pred - pos_y_pred + 0.3))
            loss = ctr_loss + self.losses

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(pos_y, pos_y_pred)
        self.compiled_metrics.update_state(neg_y, neg_y_pred)
        return {m.name: m.result() for m in self.metrics}


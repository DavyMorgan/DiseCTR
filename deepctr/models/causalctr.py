# -*- coding:utf-8 -*-
from deepctr.models.afm import AFM
from itertools import chain

import tensorflow as tf
from tensorflow import keras

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN, LNN
from ..layers.interaction import InteractingLayer, AttentionLayer, ProbAttention
from ..layers.interaction import FM, FMVec, AFMLayer
from ..layers.utils import concat_func, add_func, combined_dnn_input


def LinearPart(linear_feature_columns, l2_reg_linear=1e-5, seed=1024):

    features = build_input_features(linear_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=linear_logit)
    return model


def FMPart(dnn_feature_columns, l2_reg_embedding=1e-5, seed=1024, fm_group=[DEFAULT_GROUP_NAME]):

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    model = tf.keras.models.Model(inputs=inputs_list, outputs=fm_logit)
    return model


def FinegrainedInterestEncoderMLP(dnn_feature_columns, num_interests, dnn_hidden_units=(32, 32),
        l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu'):

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    output = []
    for i in range(num_interests):
        dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False,
            seed=seed, train_bias=False, name='interest_encoder_{}'.format(i))(dnn_input)
        dnn_out = tf.expand_dims(dnn_out, axis=1)
        output.append(dnn_out)
    output = concat_func(output, axis=-2)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FinegrainedInterestEncoderAutoInt(dnn_feature_columns, num_interests, att_layer_num=3, att_embedding_size=8, att_head_num=2,
            att_res=True, dnn_hidden_units=(256, 256), dnn_activation='relu', 
            l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, seed=1024, model_mode='pair'):

    use_bias = model_mode == 'point'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    output = []
    for i in range(num_interests):
        att_input = concat_func(sparse_embedding_list, axis=1)
        for _ in range(att_layer_num):
            att_input = InteractingLayer(
                att_embedding_size, att_head_num, att_res)(att_input)
        att_output = tf.keras.layers.Flatten()(att_input)
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, train_bias=use_bias)(dnn_input)
        stack_out = tf.keras.layers.Concatenate()([att_output, deep_out])
        stack_out = tf.expand_dims(stack_out, axis=1)
        output.append(stack_out)
    output = concat_func(output, axis=-2)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FinegrainedInterestEncoderAFN(dnn_feature_columns, num_interests, lnn_dim, dnn_hidden_units=(256, 256), lnn_activation='relu',
            l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_activation='relu', dnn_use_bn=False, dnn_dropout=0, seed=1024, model_mode='pair'):

    use_bias = model_mode == 'point'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    lnn_input = concat_func(sparse_embedding_list, axis=1)

    output = []
    for i in range(num_interests):
        lnn_output = LNN(lnn_dim, lnn_activation, l2_reg=l2_reg_dnn, seed=seed, use_bias=use_bias)(lnn_input)
        lnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, train_bias=use_bias)(lnn_output)
        lnn_output = tf.expand_dims(lnn_output, axis=1)
        output.append(lnn_output)
    output = concat_func(output, axis=-2)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


def FinegrainedInterestEncoderInformer(dnn_feature_columns, num_interests, att_layer_num=3, att_embedding_size=8, att_head_num=2,
            dnn_hidden_units=(256, 256), dnn_activation='relu', l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, seed=1024,
            model_mode='pair'):

    use_bias = model_mode == 'point'

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    output = []
    for i in range(num_interests):
        att_input = concat_func(sparse_embedding_list, axis=1)
        for _ in range(att_layer_num):
            att_input = AttentionLayer(ProbAttention(False, 1), att_embedding_size, att_head_num)([att_input, att_input, att_input])
        att_output = tf.keras.layers.Flatten()(att_input)
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, train_bias=use_bias)(dnn_input)
        stack_out = tf.keras.layers.Concatenate()([att_output, deep_out])
        stack_out = tf.expand_dims(stack_out, axis=1)
        output.append(stack_out)
    output = concat_func(output, axis=-2)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


class Projector(keras.layers.Layer):

    def __init__(self, num_interests):

        super(Projector, self).__init__()
        self.num_interests = num_interests
        self.softmax = tf.keras.layers.Softmax(axis=-2)

    def build(self, input_shape):

        self.prototype = self.add_weight(
            shape=(self.num_interests, input_shape[-1]),
            initializer='glorot_normal',
            trainable=True,
            name='prototype'
        )

    def call(self, inputs):

        normed_interests = tf.math.l2_normalize(inputs, axis=-1)
        normed_prototype = tf.math.l2_normalize(self.prototype, axis=-1)

        cosine_sim = tf.matmul(normed_prototype, normed_interests, transpose_b=True)
        cosine_sim = self.softmax(cosine_sim)

        projected_interests = tf.matmul(cosine_sim, inputs)

        return projected_interests


class PointDisentangler(keras.layers.Layer):

    def __init__(self, loss_weight):

        super(PointDisentangler, self).__init__()
        self.loss_weight = loss_weight

    def call(self, inputs):

        interests = inputs

        normed_interests = tf.math.l2_normalize(interests, axis=-1)

        cosine_sim = tf.matmul(normed_interests, normed_interests, transpose_b=True)

        discrepancy = tf.linalg.band_part(cosine_sim, 0, -1) - tf.linalg.band_part(cosine_sim, 0, 0)
        discrepancy_loss = tf.math.reduce_sum(discrepancy)
        self.add_loss(self.loss_weight*discrepancy_loss)

        return inputs


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


class PairDisentangler(keras.layers.Layer):

    def __init__(self, num_interests, diff_interests, ood_mode, discrepancy_loss_weight, weak_loss_weight, weak_supervision=True, add_average=True, num_color=None, add_adv=False, seed=1024):

        super(PairDisentangler, self).__init__()
        self.num_interests = num_interests
        self.diff_interests = diff_interests
        self.ood_mode = ood_mode
        self.discrepancy_loss_weight = discrepancy_loss_weight
        self.weak_supervision = weak_supervision
        if ood_mode == 'easy':
            self.weak_loss_weight = weak_loss_weight
            self.num_color = num_color
            self.add_adv = add_adv
            self.add_average = add_average
            self.weak_linear = tf.keras.layers.Dense(num_color, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))
            self.weak_criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='weak_cross_entropy')

    def call(self, inputs):

        if self.weak_supervision:
            if self.ood_mode == 'hard':
                pos_interests, neg_interests = inputs
                mean_interests = 0.5*pos_interests + 0.5*neg_interests
                if self.diff_interests > 0:
                    distance = tf.math.reduce_sum(tf.math.squared_difference(pos_interests, neg_interests), axis=-1)
                    values, _ = tf.math.top_k(distance, k=self.diff_interests)
                    thres_distance = tf.expand_dims(values[:,-1], -1)
                    mask = tf.expand_dims(tf.less(distance, thres_distance), -1)

                    pos_interests = tf.where(mask, mean_interests, pos_interests)
                    neg_interests = tf.where(mask, mean_interests, neg_interests)
                else:
                    pos_interests = mean_interests
                    neg_interests = mean_interests
            elif self.ood_mode == 'easy':
                pos_interests, neg_interests, pos_color, neg_color = inputs
                pos_color = tf.expand_dims(pos_color, -1)
                neg_color = tf.expand_dims(neg_color, -1)
                mask = tf.math.equal(pos_color, neg_color)
                pos_unsupervised_interests, pos_weaksupervised_interests = tf.split(pos_interests, [self.diff_interests, self.num_interests - self.diff_interests], 1)
                neg_unsupervised_interests, neg_weaksupervised_interests = tf.split(neg_interests, [self.diff_interests, self.num_interests - self.diff_interests], 1)
                if self.add_average:
                    mean_weaksupervised_interests = 0.5*pos_weaksupervised_interests + 0.5*neg_weaksupervised_interests
                    pos_weaksupervised_interests = tf.where(mask, mean_weaksupervised_interests, pos_weaksupervised_interests)
                    neg_weaksupervised_interests = tf.where(mask, mean_weaksupervised_interests, neg_weaksupervised_interests)
                pos_interests = tf.concat([pos_unsupervised_interests, pos_weaksupervised_interests], 1)
                neg_interests = tf.concat([neg_unsupervised_interests, neg_weaksupervised_interests], 1)

                if self.add_adv:
                    pos_unsupervised_interests_reverse = GradientReversalLayer()(pos_unsupervised_interests)
                    neg_unsupervised_interests_reverse = GradientReversalLayer()(neg_unsupervised_interests)
                    pos_interests_reverse = tf.concat([pos_unsupervised_interests_reverse, pos_weaksupervised_interests], 1)
                    neg_interests_reverse = tf.concat([neg_unsupervised_interests_reverse, neg_weaksupervised_interests], 1)
                    pos_logit = self.weak_linear(pos_interests_reverse)
                    neg_logit = self.weak_linear(neg_interests_reverse)
                    pos_label = tf.tile(tf.squeeze(pos_color, [-1]), [1, self.num_interests])
                    neg_label = tf.tile(tf.squeeze(neg_color, [-1]), [1, self.num_interests])
                    pos_weak_loss = self.weak_criterion(pos_label, pos_logit)
                    neg_weak_loss = self.weak_criterion(neg_label, neg_logit)
                    weak_loss = pos_weak_loss + neg_weak_loss
                    self.add_loss(self.weak_loss_weight*weak_loss)
                else:
                    pos_weak_logit = self.weak_linear(pos_weaksupervised_interests)
                    neg_weak_logit = self.weak_linear(neg_weaksupervised_interests)
                    pos_weak_loss = self.weak_criterion(pos_color, pos_weak_logit)
                    neg_weak_loss = self.weak_criterion(neg_color, neg_weak_logit)
                    weak_loss = pos_weak_loss + neg_weak_loss
                    self.add_loss(self.weak_loss_weight*weak_loss)
        else:
            if self.ood_mode == 'hard':
                pos_interests, neg_interests = inputs
            elif self.ood_mode == 'easy':
                pos_interests, neg_interests, _, _ = inputs

        normed_pos_interests = tf.math.l2_normalize(pos_interests, axis=-1)
        normed_neg_interests = tf.math.l2_normalize(neg_interests, axis=-1)

        pos_cosine_sim = tf.matmul(normed_pos_interests, normed_pos_interests, transpose_b=True)
        neg_cosine_sim = tf.matmul(normed_neg_interests, normed_neg_interests, transpose_b=True)

        pos_discrepancy = tf.linalg.band_part(pos_cosine_sim, 0, -1) - tf.linalg.band_part(pos_cosine_sim, 0, 0)
        neg_discrepancy = tf.linalg.band_part(neg_cosine_sim, 0, -1) - tf.linalg.band_part(neg_cosine_sim, 0, 0)
        discrepancy_loss = tf.math.reduce_sum(pos_discrepancy) + tf.math.reduce_sum(neg_discrepancy)

        self.add_loss(self.discrepancy_loss_weight*discrepancy_loss)

        return pos_interests, neg_interests


class CausalClickPredictor(keras.layers.Layer):

    def __init__(self, use_bias=False, task='binary', num_heads=4, seed=1024, add_linear_fm=False):

        super(CausalClickPredictor, self).__init__()
        self.num_heads = num_heads
        self.add_linear_fm = add_linear_fm
        self.att = tf.keras.layers.Attention()
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(1, use_bias=use_bias, kernel_initializer=tf.keras.initializers.glorot_normal(seed))
        self.activation = PredictionLayer(task, use_bias=use_bias)

    def build(self, input_shape):

        self.att_mat = self.add_weight(
            shape=(1, self.num_heads, input_shape[0][-1]),
            initializer='glorot_normal',
            trainable=True,
            name='att_mat'
        )
        super(CausalClickPredictor, self).build(input_shape)

    def call(self, inputs):

        key_value = inputs[0]
        batch_size = tf.shape(key_value)[0]
        query = tf.tile(self.att_mat, [batch_size, 1, 1])
        att_merged = self.att([query, key_value])
        att_merged = self.flatten(att_merged)
        logit = self.linear(att_merged)
        if self.add_linear_fm:
            linear_fm_logit = inputs[1]
            final_logit = logit + linear_fm_logit
        else:
            final_logit = logit
        score = self.activation(final_logit)

        return score


class PointCausalCTR(keras.Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, color_feat, flags_obj, model_mode):

        super(PointCausalCTR, self).__init__()
        self.linear = LinearPart(linear_feature_columns)
        self.fm = FMPart(dnn_feature_columns)
        self.fie = self.get_fie(dnn_feature_columns, flags_obj)
        self.add_projector = flags_obj.add_projector
        self.projector = Projector(flags_obj.num_interests)
        self.disentangler = self.get_disentangler(flags_obj, color_feat)
        self.ccp = self.get_ccp(flags_obj, color_feat)
        self.ood_mode = flags_obj.ood_mode
        if flags_obj.ood_mode == 'easy':
            self.color_name = color_feat.name
        self.model_mode = model_mode

    def get_fie(self, dnn_feature_columns, flags_obj):
        if flags_obj.base_model == 'MLP':
            fie = FinegrainedInterestEncoderMLP(dnn_feature_columns, flags_obj.num_interests,
                dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
        elif flags_obj.base_model == 'AutoInt':
            fie = FinegrainedInterestEncoderAutoInt(dnn_feature_columns, flags_obj.num_interests,
                dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), att_layer_num=flags_obj.att_layer_num)
        elif flags_obj.base_model == 'Informer':
            fie = FinegrainedInterestEncoderInformer(dnn_feature_columns, flags_obj.num_interests, att_embedding_size=flags_obj.embedding_dim,
                dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units), att_layer_num=flags_obj.att_layer_num)
        elif flags_obj.base_model == 'AFN':
            fie = FinegrainedInterestEncoderAFN(dnn_feature_columns, flags_obj.num_interests, flags_obj.lnn_dim, 
                dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
        return fie

    def get_disentangler(self, flags_obj, color_feat):

        disentangler = PointDisentangler(flags_obj.discrepancy_loss_weight)
        return disentangler

    def get_ccp(self, flags_obj, color_feat):

        ccp = CausalClickPredictor(num_heads=flags_obj.num_heads, add_linear_fm=flags_obj.add_linear_fm)
        return ccp

    def call(self, inputs, training=None):

        if self.ood_mode == 'easy':
            color = inputs[self.color_name]

        inputs_x = inputs

        interests = self.fie(inputs_x, training=training)

        if self.add_projector:
            interests = self.projector(interests)

        interests = self.disentangler(interests)

        linear_logit = tf.reshape(self.linear(inputs_x, training=training), [-1, 1])

        fm_logit = self.fm(inputs_x, training=training)

        linear_fm_logit = linear_logit + fm_logit

        score = self.ccp([interests, linear_fm_logit])

        return score


class PairCausalCTR(PointCausalCTR):
    def __init__(self, linear_feature_columns, dnn_feature_columns, color_feat, flags_obj, model_mode):

        super(PairCausalCTR, self).__init__(linear_feature_columns, dnn_feature_columns, color_feat, flags_obj, model_mode)

    def get_disentangler(self, flags_obj, color_feat):

        if flags_obj.ood_mode == 'easy':
            num_color = color_feat.vocabulary_size
        else:
            num_color = None
        disentangler = PairDisentangler(flags_obj.num_interests, flags_obj.diff_interests, flags_obj.ood_mode, 
                                        flags_obj.discrepancy_loss_weight, flags_obj.weak_loss_weight,
                                        weak_supervision=flags_obj.weak_supervision, add_average=flags_obj.add_average,
                                        num_color=num_color, add_adv=flags_obj.add_adv)
        return disentangler

    def call(self, inputs, training=None):

        pos_x, neg_x = inputs
        if self.ood_mode == 'easy':
            pos_color = pos_x[self.color_name]
            neg_color = neg_x[self.color_name]
            pos_color = tf.expand_dims(pos_color, -1)
            neg_color = tf.expand_dims(neg_color, -1)

        input_pos_x = pos_x
        input_neg_x = neg_x

        pos_interests = self.fie(input_pos_x, training=training)
        neg_interests = self.fie(input_neg_x, training=training)

        if self.add_projector:
            pos_interests = self.projector(pos_interests)
            neg_interests = self.projector(neg_interests)

        if self.ood_mode == 'hard':
            pos_interests, neg_interests = self.disentangler([pos_interests, neg_interests])
        elif self.ood_mode == 'easy':
            pos_interests, neg_interests = self.disentangler([pos_interests, neg_interests, pos_color, neg_color])

        pos_linear_logit = tf.reshape(self.linear(input_pos_x, training=training), [-1, 1])
        neg_linear_logit = tf.reshape(self.linear(input_neg_x, training=training), [-1, 1])

        pos_fm_logit = self.fm(input_pos_x, training=training)
        neg_fm_logit = self.fm(input_neg_x, training=training)

        pos_linear_fm_logit = pos_linear_logit + pos_fm_logit
        neg_linear_fm_logit = neg_linear_logit + neg_fm_logit

        pos_score = self.ccp([pos_interests, pos_linear_fm_logit])
        neg_score = self.ccp([neg_interests, neg_linear_fm_logit])

        return pos_score, neg_score

    def train_step(self, data):
        [pos_x, neg_x], [pos_y, neg_y] = data

        with tf.GradientTape() as tape:
            pos_score, neg_score = self([pos_x, neg_x], training=True)

            ctr_loss = tf.math.reduce_sum(tf.math.maximum(0.0, neg_score - pos_score + 0.3))
            loss = ctr_loss + self.losses

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(pos_y, pos_score)
        self.compiled_metrics.update_state(neg_y, neg_score)
        return {m.name: m.result() for m in self.metrics}

    def point_inference(self, x, return_interests=False):

        input_x = x
        interests = self.fie(input_x, training=False)
        if self.add_projector:
            interests = self.projector(interests)
        linear_logit = tf.reshape(self.linear(input_x, training=False), [-1, 1])
        fm_logit = self.fm(input_x, training=False)
        linear_fm_logit = linear_logit + fm_logit
        score = self.ccp([interests, linear_fm_logit], training=False)

        if not return_interests:
            return score
        else:
            return score, interests

    def test_step(self, data):

        x, y = data
        score = self.point_inference(x)

        self.compiled_metrics.update_state(y, score)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):

        x, = data
        if self.model_mode == 'index':
            score, interests = self.point_inference(x, return_interests=True)
            return score, interests
        else:
            score = self.point_inference(x)
            return score


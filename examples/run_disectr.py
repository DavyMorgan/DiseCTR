from absl import app
from absl import flags

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

import tensorflow as tf
from tensorflow import keras

import os
import sys
sys.path.append('../')

from deepctr.layers.utils import gauc_score
from deepctr.feature_column import get_feature_names
from examples.utils import load_yaml, auto_dataset_setup, get_mail_master, get_data, get_feature_target, split_data, PairSequence, protect_message, get_model, get_callbacks

import setproctitle
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'debug', 'Exp name.')
flags.DEFINE_string('model', 'DeepFM', 'Recommendation model.')
flags.DEFINE_boolean('run_eagerly', False, 'Whether to debug with eager mode.')
flags.DEFINE_boolean('test_new_user', False, 'Whether to set test user embedding as random vector.')
flags.DEFINE_boolean('has_uid', True, 'Whether the dataset has uid feature or not.')
flags.DEFINE_boolean('remove_color_feature', False, 'Whether to remove color feature.')
flags.DEFINE_enum('mode', 'train', ['train', 'test', 'finetune', 'index'], 'Whether to load checkpoint and evaluation.')
flags.DEFINE_boolean('analyze_model', False, 'Whether to analyze model.')
flags.DEFINE_enum('ood_mode', 'easy', ['easy', 'hard', 'no'], 'OOD mode.')
flags.DEFINE_multi_float('multi_ood_irm_e', [0.4], 'Multiple ood irm e values for fast test.')
flags.DEFINE_float('irm_e', 0.6, 'E value for IRM.')
flags.DEFINE_string('save_path', './', 'Save path if test_ckpt is True.')
flags.DEFINE_enum('dataset', 'kuaishou', ['amazon', 'kuaishou', 'wechat'], 'Dataset.')
flags.DEFINE_enum('train_mode', 'pair', ['pair', 'point'], 'Pairwise of pointwise training mode.')
flags.DEFINE_string('base_model', 'Informer', 'Basemodel for causalctr.')
flags.DEFINE_boolean('add_linear_fm', False, 'Whether to add linear and fm part.')
flags.DEFINE_boolean('add_projector', True, 'Whether to add projector.')
flags.DEFINE_boolean('add_average', True, 'Whether to add average embedding for weak supervision.')
flags.DEFINE_boolean('add_adv', True, 'Whether to add gradient reversal layer for weak supervision.')
flags.DEFINE_integer('num_interests', 4, 'The number of finegrained interests.')
flags.DEFINE_integer('diff_interests', 3, 'The number of different finegrained interests.')
flags.DEFINE_integer('num_heads', 8, 'The number of attention heads.')
flags.DEFINE_float('finetune_lr', 0.0001, 'Initial learning rate for finetune.')
flags.DEFINE_float('finetune_sample_frac', 0.01, 'Fraction of training data to finetune.')
flags.DEFINE_boolean('freeze_embedding', False, 'Whether to freeze embeddings for finetune.')
flags.DEFINE_float('discrepancy_loss_weight', 0.1, 'Discrepancy loss weight.')
flags.DEFINE_boolean('weak_supervision', True, 'Whether to add weak supervision on disentanglement.')
flags.DEFINE_float('weak_loss_weight', 0.1, 'Weak supervision loss weight.')
flags.DEFINE_float('sample_frac', 0.5, 'Fraction of sampled IID and OOD test data.')
flags.DEFINE_integer('embedding_dim', 16, 'Embedding size.')
flags.DEFINE_integer('dnn_hidden_units', 64, 'Hidden units of DNN.')
flags.DEFINE_integer('att_layer_num', 2, 'The number of attention layers.')
flags.DEFINE_integer('lnn_dim', 16, 'LNN dimension for AFN.')
flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
flags.DEFINE_integer('epochs', 100, 'The number of epochs.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')
flags.DEFINE_boolean('enable_mail_service', True, 'Whether to e-mail yourself after each run.')
flags.DEFINE_boolean('repeat_experiment', False, 'Whether to repeat experiments with outside shell script.')
flags.DEFINE_string('mail_pass', 'xxx', 'Password for e-mail service provided by shell read input.')
flags.DEFINE_string('mail_host', 'xxx', 'Host for email service.')
flags.DEFINE_string('mail_username', 'xxx', 'Username for email service.')
flags.DEFINE_string('receiver_address', 'xxx', 'E-mail receiver address.')
flags.DEFINE_integer('mail_port', 587, 'Port for email service.')

flags.register_validator('name',
    lambda value: value != 'debug',
    message='Exp name not informative! Provide a valid exp name!')


def train_model(flags_obj, cfg, save_path, model, feature_names, data_splits, model_input, target, transfer_target):

    callbacks = get_callbacks(flags_obj, save_path)
    train_model_input = model_input[0]
    train = data_splits[0]
    val_model_input = model_input[1]
    val = data_splits[1]

    if flags_obj.mode == 'finetune':
        optimizer = tf.keras.optimizers.Adam(learning_rate=flags_obj.finetune_lr)
        label = transfer_target
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        label = target

    if flags_obj.train_mode == 'pair':
        sequence = PairSequence(train, feature_names, label, flags_obj.batch_size)
        model.compile(optimizer, metrics=['binary_crossentropy', keras.metrics.AUC(name='auc')], run_eagerly=flags_obj.run_eagerly)
        history = model.fit(sequence, epochs=flags_obj.epochs, verbose=2, validation_batch_size=flags_obj.batch_size,
                            validation_data=(val_model_input, val[label].values), callbacks=callbacks,
                            workers=4, use_multiprocessing=False)
    else:
        model.compile(optimizer, "binary_crossentropy", metrics=['binary_crossentropy', keras.metrics.AUC(name='auc')], run_eagerly=flags_obj.run_eagerly)
        history = model.fit(train_model_input, train[label].values,
                            batch_size=flags_obj.batch_size, epochs=flags_obj.epochs, verbose=2, 
                            validation_data=(val_model_input, val[label].values), callbacks=callbacks)

    return history


def test_model(flags_obj, cfg, model, data_splits, model_input, target, transfer_target):

    ood_pred_ans_list = []
    OOD_AUC_list = []
    OOD_GAUC_list = []
    OOD_logloss_list = []

    if flags_obj.has_uid:
        uid_feature_name = cfg['data'][flags_obj.dataset]['uid_feature_name']
    (_, _, iid_test, ood_tests) = data_splits
    (_, _, iid_test_model_input, ood_test_model_inputs) = model_input
    iid_pred_ans = model.predict(iid_test_model_input, batch_size=flags_obj.batch_size)
    for ood_test_model_input in ood_test_model_inputs:
        ood_pred_ans = model.predict(ood_test_model_input, batch_size=flags_obj.batch_size)
        ood_pred_ans_list.append(ood_pred_ans)

    IID_AUC = round(roc_auc_score(iid_test[target].values, iid_pred_ans), 4)
    if flags_obj.has_uid:
        IID_GAUC = round(gauc_score(iid_test[uid_feature_name].values.reshape(-1), iid_test[target].values.reshape(-1), iid_pred_ans.reshape(-1)), 4)
    IID_logloss = round(log_loss(iid_test[target].values, iid_pred_ans, eps=1e-7), 4)
    if flags_obj.ood_mode == 'easy':
        for i in range(len(flags_obj.multi_ood_irm_e)):
            ood_test = ood_tests[i]
            ood_pred_ans = ood_pred_ans_list[i]
            OOD_AUC = round(roc_auc_score(ood_test[target].values, ood_pred_ans), 4)
            OOD_AUC_list.append(OOD_AUC)
            OOD_GAUC = round(gauc_score(ood_test[uid_feature_name].values.reshape(-1), ood_test[target].values.reshape(-1), ood_pred_ans.reshape(-1)), 4)
            OOD_GAUC_list.append(OOD_GAUC)
            OOD_logloss = round(log_loss(ood_test[target].values, ood_pred_ans, eps=1e-7), 4)
            OOD_logloss_list.append(OOD_logloss)
        print("IID test AUC", IID_AUC)
        print("IID test GAUC", IID_GAUC)
        print("IID test LogLoss", IID_logloss)
        for i, ood_irm_e in enumerate(flags_obj.multi_ood_irm_e):
            print('OOD IRM E = {}'.format(ood_irm_e))
            print("OOD test AUC", OOD_AUC_list[i])
            print("OOD test GAUC", OOD_GAUC_list[i])
            print("OOD test LogLoss", OOD_logloss_list[i])

        result = {
            'IID AUC': IID_AUC,
            'IID GAUC': IID_GAUC,
            'IID logloss': IID_logloss,
            'OOD AUC': OOD_AUC_list,
            'OOD GAUC': OOD_GAUC_list,
            'OOD logloss': OOD_logloss_list
        }
    elif flags_obj.ood_mode == 'hard':
        OOD_AUC = round(roc_auc_score(iid_test[transfer_target].values, iid_pred_ans), 4)
        OOD_GAUC = round(gauc_score(iid_test[uid_feature_name].values.reshape(-1), iid_test[transfer_target].values.reshape(-1), iid_pred_ans.reshape(-1)), 4)
        OOD_logloss = round(log_loss(iid_test[transfer_target].values, iid_pred_ans, eps=1e-7), 4)

        print("IID test AUC", IID_AUC)
        print("IID test GAUC", IID_GAUC)
        print("IID test LogLoss", IID_logloss)
        print("OOD test AUC", OOD_AUC)
        print("OOD test GAUC", OOD_GAUC)
        print("OOD test LogLoss", OOD_logloss)

        result = {
            'IID AUC': IID_AUC,
            'IID GAUC': IID_GAUC,
            'IID logloss': IID_logloss,
            'OOD AUC': OOD_AUC,
            'OOD GAUC': OOD_GAUC,
            'OOD logloss': OOD_logloss
        }
    elif flags_obj.ood_mode == 'no':
        print("IID test AUC", IID_AUC)
        if flags_obj.has_uid:
            print("IID test GAUC", IID_GAUC)
        print("IID test LogLoss", IID_logloss)

        if flags_obj.has_uid:
            result = {
                'IID AUC': IID_AUC,
                'IID GAUC': IID_GAUC,
                'IID logloss': IID_logloss
            }
        else:
            result = {
                'IID AUC': IID_AUC,
                'IID logloss': IID_logloss
            }


    return result


def get_index(flags_obj, cfg, model, data_splits, model_input, target):

    (train_model_input, _, _, _) = model_input
    _, index = model.predict(train_model_input, batch_size=flags_obj.batch_size, steps=2)
    return index


def refine_irm(flags_obj):

    flags_obj.irm_e = flags_obj.irm_e/10
    flags_obj.multi_ood_irm_e = [irm_e/10 for irm_e in flags_obj.multi_ood_irm_e]


def main(argv):

    flags_obj = FLAGS

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags_obj.gpu_id)

    if flags_obj.irm_e > 1:
        refine_irm(flags_obj)

    cfg = load_yaml('./config.yaml')
    auto_dataset_setup(flags_obj, cfg)
    if flags_obj.enable_mail_service:
        mail_master = get_mail_master(flags_obj, cfg)

    #read data
    data = get_data(flags_obj, cfg)

    #transform data into feature columns
    data, fixlen_feature_columns, target, transfer_target = get_feature_target(flags_obj, cfg, data)
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    #generate input data for model
    data_splits, model_input = split_data(flags_obj, cfg, data, feature_names, target)

    #define Model,train,predict and evaluate
    model = get_model(flags_obj, cfg, linear_feature_columns, dnn_feature_columns)

    if flags_obj.mode in ['train', 'finetune']:

        if flags_obj.mode == 'finetune':
            latest = tf.train.latest_checkpoint(os.path.join(flags_obj.save_path, 'ckpt'))
            print('Loading latest checkpoint at: {}'.format(latest))

            model.load_weights(latest).expect_partial()
            if flags_obj.freeze_embedding:
                if flags_obj.model != 'CausalCTR':
                    for layer in model.layers:
                        if isinstance(layer, tf.keras.layers.Embedding):
                            layer.trainable = False
                else:
                    for layer in model.fie.layers:
                        if isinstance(layer, tf.keras.layers.Embedding):
                            layer.trainable = False
            if flags_obj.analyze_model:
                model.trainable = False

        save_path = os.path.join(cfg['train']['save_path'][flags_obj.dataset], 
                                 flags_obj.dataset + '-' + flags_obj.model + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_path)
        model_plot_path = os.path.join(save_path, 'model.png')
        keras.utils.plot_model(model, model_plot_path, show_shapes=True)
        print('Model plot at: {}'.format(model_plot_path))
        history = train_model(flags_obj, cfg, save_path, model, feature_names, data_splits, model_input, target, transfer_target)
        if flags_obj.analyze_model:
            model.summary()
            att_mat = model.ccp.get_weights()[0]
            np.save('analyze_att_after_finetune0.5.npy', att_mat)
            print(att_mat.shape)
            exit()
        result = test_model(flags_obj, cfg, model, data_splits, model_input, target, transfer_target)
        latest = tf.train.latest_checkpoint(os.path.join(save_path, 'ckpt'))
        print('Latest checkpoint at: {}'.format(latest))
    else:
        latest = tf.train.latest_checkpoint(os.path.join(flags_obj.save_path, 'ckpt'))
        print('Loading latest checkpoint at: {}'.format(latest))

        model.load_weights(latest).expect_partial()
        if flags_obj.mode == 'test':
            result = test_model(flags_obj, cfg, model, data_splits, model_input, target, transfer_target)
        else:
            index = get_index(flags_obj, cfg, model, data_splits, model_input, target)
            np.save(cfg['data'][flags_obj.dataset]['index_file_name'], index)
            print(index.shape)
            exit()


    if flags_obj.mode != 'train':
        train_auc, val_auc = 0.0, 0.0
    else:
        train_auc, val_auc = map(float, latest.split('_auc_')[1].split('_valauc_'))
    result['Train AUC'] = train_auc
    result['Val AUC'] = val_auc

    if flags_obj.enable_mail_service:
        mail_master.send_mail(flags_obj, cfg, result, latest)

    protect_message()


if __name__ == "__main__":

    app.run(main)

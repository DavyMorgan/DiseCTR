import yaml
import os
import socket
import time
import random
import getpass
import smtplib
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM, NFM, AFM, LibFM, WDL, MLP, xDeepFM, AutoInt, Informer, AFN, DESTINE, PairCausalCTR, PointCausalCTR
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import tensorflow as tf
from tensorflow import keras


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def auto_dataset_setup(flags_obj, cfg):

    if flags_obj.dataset == 'amazon':
        cfg['data']['amazon']['file_path'] = cfg['data']['amazon']['file_path']
        cfg['train']['save_path']['amazon'] = cfg['train']['save_path']['amazon']
    elif flags_obj.dataset == 'kuaishou':
        if flags_obj.ood_mode == 'easy':
            cfg['data']['kuaishou']['file_path'] = cfg['data']['kuaishou']['file_path_easy']
        elif flags_obj.ood_mode == 'hard':
            cfg['data']['kuaishou']['file_path'] = cfg['data']['kuaishou']['file_path_hard']
    elif flags_obj.dataset == 'wechat':
        cfg['train']['save_path']['wechat'] = cfg['train']['save_path']['wechat']
        if flags_obj.ood_mode == 'easy':
            cfg['data']['wechat']['file_path'] = cfg['data']['wechat']['file_path_easy']
        elif flags_obj.ood_mode == 'hard':
            cfg['data']['wechat']['file_path'] = cfg['data']['wechat']['file_path_hard']


class MailMaster(object):

    def __init__(self, flags_obj, cfg):

        username = flags_obj.mail_username
        receiver = flags_obj.receiver_address
        self.mail_host = flags_obj.mail_host
        self.port = flags_obj.mail_port
        self.mail_user = username
        self.get_mail_pass(flags_obj, cfg)
        self.sender = username
        self.receivers = [receiver]

        self.metrics = cfg['train']['metrics']

    def get_mail_pass(self, flags_obj, cfg):

        if flags_obj.repeat_experiment:
            connect_succeed = self.connect_mail(cfg, flags_obj.mail_pass)
            if not connect_succeed:
                quit()
            else:
                self.mail_pass = flags_obj.mail_pass
        else:
            self.mail_pass = self.get_mail_pass_from_prompt(cfg)

    def get_mail_pass_from_prompt(self, cfg):

        connect_succeed = False
        max_connect_time = 10
        count = 0
        while not connect_succeed:
            count = count + 1
            if count > max_connect_time:
                raise ValueError("Wrong password for too many times.")
            mail_pass = getpass.getpass('Input mail password:')
            connect_succeed = self.connect_mail(cfg, mail_pass)

        return mail_pass

    def connect_mail(self, cfg, mail_pass):

        connect_succeed = False
        try:
            if socket.gethostname() == cfg['const']['kuaishou_machine_hostname']:
                smtpObj = smtplib.SMTP()
            else:
                smtpObj = smtplib.SMTP_SSL(self.mail_host)
            smtpObj.connect(self.mail_host, self.port)
            login_status = smtpObj.login(self.mail_user, mail_pass) 
            if login_status[0] >= 200 and login_status[0] < 300:
                connect_succeed = True
                print('mail login success')
                time.sleep(2)
            smtpObj.quit() 
        except smtplib.SMTPException as e:
            print('error',e)
            connect_succeed = False

        return connect_succeed

    def get_table_head(self, res, irm_e):

        if irm_e == 'IID':
            table_head = ''.join(['<th>{}</th>'.format(metric) for metric in self.metrics if metric in res and 'OOD' not in metric])
        else:
            table_head = ''.join(['<th>{}</th>'.format(metric) for metric in self.metrics if metric in res and 'OOD' in metric])

        return table_head

    def get_table_head_hard(self, res):

        table_head = ''.join(['<th>{}</th>'.format(metric) for metric in self.metrics if metric in res])

        return table_head

    def get_res(self, res, metric, index):

        if 'OOD' not in metric:
            return res[metric]
        else:
            return res[metric][index]

    def get_table_res(self, res, index):

        if index is None:
            table_res = ''.join(['<th>{:.4f}</th>'.format(self.get_res(res, metric, index)) for metric in self.metrics if metric in res and 'OOD' not in metric])
        else:
            table_res = ''.join(['<th>{:.4f}</th>'.format(self.get_res(res, metric, index)) for metric in self.metrics if metric in res and 'OOD' in metric])

        return table_res

    def get_table_res_hard(self, res):

        table_res = ''.join(['<th>{:.4f}</th>'.format(res[metric]) for metric in self.metrics if metric in res])

        return table_res

    def get_table(self, res, irm_e, index):

        table_head = self.get_table_head(res, irm_e)
        table_res = self.get_table_res(res, index)
        html = """\
                <p>
                   irm_e: {}<br>
                </p>
                <p>
                    <table border="1">
                        <tr>
                            {}
                        </tr>
                        <tr>
                            {}
                        </tr>
                    </table>
                </p>
        """.format(irm_e, table_head, table_res)
        return html

    def get_table_hard(self, res):

        table_head = self.get_table_head_hard(res)
        table_res = self.get_table_res_hard(res)
        html = """\
                <p>
                    <table border="1">
                        <tr>
                            {}
                        </tr>
                        <tr>
                            {}
                        </tr>
                    </table>
                </p>
        """.format(table_head, table_res)
        return html

    def send_mail(self, flags_obj, cfg, res, ckpt):

        html = """\
            <html>
            <body>
                <p>
                    Exp name: {}<br>
                </p>
        """.format(flags_obj.name)

        if flags_obj.ood_mode == 'easy':
            table = self.get_table(res, 'IID', None)
            html = html + table

            for index, irm_e in enumerate(flags_obj.multi_ood_irm_e):
                table = self.get_table(res, irm_e, index)
                html = html + table
        elif flags_obj.ood_mode == 'hard':
            table = self.get_table_hard(res)
            html = html + table

        html_end = """\
                <p>
                    Latest checkpoint at: {}<br>
                </p>
            </body>
            </html>
        """.format(ckpt)

        html = html + html_end

        message = MIMEText(html, "html", "utf-8")
        message['Subject'] = 'Exp finish: {}'.format(flags_obj.name) 
        message['From'] = self.sender 
        message['To'] = ';'.join(self.receivers)

        try:
            if socket.gethostname() == cfg['const']['kuaishou_machine_hostname']:
                smtpObj = smtplib.SMTP()
            else:
                smtpObj = smtplib.SMTP_SSL(self.mail_host)
            smtpObj.connect(self.mail_host, self.port)
            smtpObj.login(self.mail_user, self.mail_pass) 
            smtpObj.sendmail(self.sender, self.receivers, message.as_string()) 
            smtpObj.quit() 
            print('mail success')
        except smtplib.SMTPException as e:
            print('error',e)


def get_mail_master(flags_obj, cfg):

    flags_obj.mail_host = cfg['const']['mail_host']
    flags_obj.mail_port = cfg['const']['mail_port']
    flags_obj.mail_username = cfg['const']['mail_username']
    flags_obj.receiver_address = cfg['const']['mail_receiver_address']
    mail_master = MailMaster(flags_obj, cfg)
    return mail_master


def flip(x, p):

    if random.random() < p:
        return x
    else:
        return 1 - x


def get_data(flags_obj, cfg):

    data = pd.read_csv(cfg['data'][flags_obj.dataset]['file_path'], index_col=0)
    data = data[cfg['data'][flags_obj.dataset]['used_feature_columns']]

    if flags_obj.dataset == 'wechat':
        data['bgm_song_id'] = data['bgm_song_id'].astype('Int64').astype(str)
        data['bgm_singer_id'] = data['bgm_singer_id'].astype('Int64').astype(str)

    return data


def get_feature_target(flags_obj, cfg, data):

    sparse_features = cfg['data'][flags_obj.dataset]['sparse_features']
    dense_features = cfg['data'][flags_obj.dataset]['dense_features']
    target = [cfg['data'][flags_obj.dataset]['target_feature_name']]
    if flags_obj.ood_mode == 'hard':
        transfer_target = [cfg['data'][flags_obj.dataset]['transfer_target_feature_name']]
    else:
        transfer_target = target
    if flags_obj.model == 'MF':
        sparse_features = [cfg['data'][flags_obj.dataset]['uid_feature_name'], cfg['data'][flags_obj.dataset]['iid_feature_name']]
        dense_features = []
        flags_obj.model = 'LibFM'

    data[sparse_features] = data[sparse_features].fillna('-1', )
    if len(dense_features) > 0:
        data[dense_features] = data[dense_features].fillna(0, )

    #Label Encoding for sparse features,and do simple Transformation for dense features
    if flags_obj.has_uid:
        uid_feature_name = cfg['data'][flags_obj.dataset]['uid_feature_name']
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        if flags_obj.has_uid and feat == uid_feature_name:
            data[feat] = data[feat] + 1
    if len(dense_features) > 0:
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

    if len(dense_features) > 0:
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=flags_obj.embedding_dim)
                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                            for feat in dense_features]
    else:
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=flags_obj.embedding_dim) for i,feat in enumerate(sparse_features)]

    return data, fixlen_feature_columns, target, transfer_target


def sample_irm_core(data, irm_pos_neg_rate, target, seed):

    pos_data = data[data[target[0]]==1]
    neg_data = data[data[target[0]]==0]
    num_pos = len(pos_data)
    num_neg = len(neg_data)
    if num_pos/num_neg > irm_pos_neg_rate:
        pos_data = pos_data.sample(n=int(num_neg*irm_pos_neg_rate), random_state=seed)
    else:
        neg_data = neg_data.sample(n=int(num_pos/irm_pos_neg_rate), random_state=seed)
    data = pd.concat([pos_data, neg_data]).reset_index(drop=True)
    return data


def sample_irm(flags_obj, cfg, data, irm_e, target, seed):

    if flags_obj.dataset == 'amazon': # cheap < expensive
        color_order = 0
    elif flags_obj.dataset == 'kuaishou': # short > long
        color_order = 1
    elif flags_obj.dataset == 'wechat': # 0 < 1
        color_order = 0
    else:
        raise ValueError('Only support amazon and kuaishou dataset!')
    color_feature = cfg['data'][flags_obj.dataset]['color_feature_name']

    irm_pos_neg_rate = irm_e/(1 - irm_e)
    red = data[data[color_feature]==color_order]
    red_rate = irm_pos_neg_rate
    red = sample_irm_core(red, red_rate, target, seed)
    green = data[data[color_feature]==(1-color_order)]
    green_rate = 1/irm_pos_neg_rate
    green = sample_irm_core(green, green_rate, target, seed)
    data = pd.concat([red, green]).reset_index(drop=True)
    return data


def split_data(flags_obj, cfg, data, feature_names, target):

    ood_tests = []
    ood_test_model_inputs = []

    train_val_split = cfg['data'][flags_obj.dataset]['train_val_split']
    val_test_split = cfg['data'][flags_obj.dataset]['val_test_split']
    timestamp_key = cfg['data'][flags_obj.dataset]['timestamp_key']
    train = data[data[timestamp_key] < train_val_split]
    if flags_obj.mode == 'finetune':
        train = train.sample(frac=flags_obj.finetune_sample_frac, random_state=cfg['run']['seed'])
    val_test = data[data[timestamp_key] >= train_val_split]
    val = val_test[val_test[timestamp_key] < val_test_split]
    test = val_test[val_test[timestamp_key] >= val_test_split]

    if flags_obj.ood_mode == 'easy':
        if flags_obj.mode in ['train', 'test', 'index']:
            train_val_irm_e = flags_obj.irm_e
        elif flags_obj.mode == 'finetune':
            train_val_irm_e = 1 - flags_obj.irm_e
        else:
            raise ValueError('Mode {} is not supported for irm!'.format(flags_obj.mode))
        train = sample_irm(flags_obj, cfg, train, train_val_irm_e, target, cfg['run']['seed'])
        val = sample_irm(flags_obj, cfg, val, train_val_irm_e, target, cfg['run']['seed'])
        for ood_irm_e in flags_obj.multi_ood_irm_e:
            ood_test = sample_irm(flags_obj, cfg, test, ood_irm_e, target, cfg['run']['seed'])
            ood_tests.append(ood_test)
        iid_test = sample_irm(flags_obj, cfg, test, flags_obj.irm_e, target, cfg['run']['seed'])
    elif flags_obj.ood_mode == 'hard':
        ood_tests = []
        ood_test_model_inputs = []
        iid_test = test.sample(frac=flags_obj.sample_frac, random_state=cfg['run']['seed'])

    val_model_input = {name:val[name].values for name in feature_names}
    iid_test_model_input = {name:iid_test[name].values for name in feature_names}
    for ood_test in ood_tests:
        ood_test_model_input = {name:ood_test[name].values for name in feature_names}
        ood_test_model_inputs.append(ood_test_model_input)

    train_model_input = {name:train[name].values for name in feature_names}
    return (train, val, iid_test, ood_tests), (train_model_input, val_model_input, iid_test_model_input, ood_test_model_inputs)


def split_train(data, target):

    pos_data = data[data[target[0]] == 1].reset_index(drop=True)
    neg_data = data[data[target[0]] == 0].reset_index(drop=True)

    return pos_data, neg_data


class PairSequence(tf.keras.utils.Sequence):

    def __init__(self, train, feature_names, target, batch_size):

        self.pos_train, self.neg_train = split_train(train, target)
        self.feature_names = feature_names
        self.target = target
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil((len(self.pos_train) + len(self.neg_train))/float(self.batch_size)))

    def __getitem__(self, idx):

        batch_pos_x = {k:v[idx * self.batch_size:(idx + 1) * self.batch_size] for k, v in self.pos_train_model_input.items()}
        batch_neg_x = {k:v[idx * self.batch_size:(idx + 1) * self.batch_size] for k, v in self.neg_train_model_input.items()}

        batch_pos_y = self.pos_y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_neg_y = self.neg_y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_pos_x, batch_neg_x], [batch_pos_y, batch_neg_y]

    def on_epoch_end(self):

        self.pos_pair = self.neg_train.sample(n=len(self.pos_train), replace=True)
        self.neg_pair = self.pos_train.sample(n=len(self.neg_train), replace=True)

        self.plain_pos_train = pd.concat([self.pos_train, self.neg_pair]).reset_index(drop=True)
        self.plain_neg_train = pd.concat([self.pos_pair, self.neg_train]).reset_index(drop=True)

        self.pos_train_model_input = {name:self.plain_pos_train[name].values for name in self.feature_names}
        self.neg_train_model_input = {name:self.plain_neg_train[name].values for name in self.feature_names}

        self.pos_y = self.plain_pos_train[self.target].values
        self.neg_y = self.plain_neg_train[self.target].values


def protect_message():

    for _ in range(4):
        print('------------------------------')


def get_model(flags_obj, cfg, linear_feature_columns, dnn_feature_columns):

    if flags_obj.remove_color_feature:
        color_feature = cfg['data'][flags_obj.dataset]['color_feature_name']
        linear_feature_columns = list(filter(lambda x: x.name != color_feature, linear_feature_columns))
        dnn_feature_columns = list(filter(lambda x: x.name != color_feature, dnn_feature_columns))

    model_mode = flags_obj.mode if flags_obj.mode == 'index' else flags_obj.train_mode
    if flags_obj.model == 'DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', 
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'NFM':
        model = NFM(linear_feature_columns, dnn_feature_columns, task='binary',
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'LibFM':
        model = LibFM(linear_feature_columns, dnn_feature_columns, task='binary', model_mode=model_mode)
    elif flags_obj.model == 'WDL':
        model = WDL(linear_feature_columns, dnn_feature_columns, task='binary', 
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'AutoInt':
        model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', att_layer_num=flags_obj.att_layer_num,
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'AFN':
        model = AFN(linear_feature_columns, dnn_feature_columns, task='binary',
            model_mode=model_mode, lnn_dim=flags_obj.lnn_dim,
            dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'DESTINE':
        model = DESTINE(linear_feature_columns, dnn_feature_columns, task='binary', att_layer_num=flags_obj.att_layer_num,
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units))
    elif flags_obj.model == 'Informer':
        model = Informer(dnn_feature_columns, task='binary', att_layer_num=flags_obj.att_layer_num,
            model_mode=model_mode, dnn_hidden_units=(flags_obj.dnn_hidden_units, flags_obj.dnn_hidden_units),
            att_embedding_size=flags_obj.embedding_dim)
    elif flags_obj.model == 'CausalCTR':
        if flags_obj.ood_mode == 'easy':
            color_feature = list(filter(lambda x: x.name == cfg['data'][flags_obj.dataset]['color_feature_name'], dnn_feature_columns))
            color_feature = color_feature[0]
        else:
            color_feature = None
        input_features = dnn_feature_columns
        if model_mode == 'pair':
            model = PairCausalCTR(input_features, input_features, color_feature, flags_obj, model_mode=model_mode)
        elif model_mode == 'point':
            model = PointCausalCTR(input_features, input_features, color_feature, flags_obj, model_mode=model_mode)
        elif model_mode == 'index':
            model = PairCausalCTR(input_features, input_features, color_feature, flags_obj, model_mode=model_mode)
    else:
        raise ValueError('Model {} not supported!'.format(flags_obj.model))

    return model


def get_callbacks(flags_obj, save_path):

    ckpt_path = os.path.join(save_path, 'ckpt')
    log_path = os.path.join(save_path, 'log')
    tb_path = os.path.join(save_path, 'tb')
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    os.makedirs(tb_path)

    es = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)
    if flags_obj.mode != 'train':
        filepath = os.path.join(ckpt_path, "model")
    else:
        filepath = os.path.join(ckpt_path, "model_epoch_{epoch}_auc_{auc}_valauc_{val_auc}")
    ckpt = keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="val_auc", mode="max", save_weights_only=True, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.1, mode='max', patience=1, min_lr=0.00001)

    csv_log = keras.callbacks.CSVLogger(filename=os.path.join(log_path, 'training.log'))
    tb = keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=0, write_images=False, embeddings_freq=0)

    return [es, ckpt, csv_log, tb, reduce_lr]

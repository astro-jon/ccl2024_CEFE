import logging
import os
import random
import time
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)

    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))

    return log


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_csv_data(data_path):
    min_max = MinMaxScaler(feature_range=(0, 1))
    data_df = pd.read_csv(data_path)
    data_vec = data_df.values
    data_vec = np.hstack((data_vec[:, 0].reshape(-1, 1), min_max.fit_transform(data_vec[:, 1:])))
    return data_vec


def feature_to_tensor(feature, text_id):
    feature_list = []
    for i in text_id:
        for j in feature:
            if i.item() == int(j[0]):
                feature_list.append(j[34:].tolist())
                break
        else:
            print(i)
    feature_torch = torch.Tensor(feature_list)
    return feature_torch


def gaussian_feature_to_tensor(feature, text):
    feature = feature.to_numpy()
    feature_list = []
    for i in text:
        for j in feature:
            if i == j[1]:
                feature_list.append(j[2:feature.shape[1]].tolist())
                break
        else:
            print(i)
    feature_torch = torch.Tensor(feature_list)
    return feature_torch


def read_data(data_path):
    data_df = pd.read_csv(data_path, header=None)
    data_vec = data_df.values[:, 1:]
    min_max = MinMaxScaler(feature_range=(0, 1))
    data_vec = min_max.fit_transform(data_vec)
    item_id_df = data_df[[0]]
    item_id_df.rename(columns={0: 'item_id'}, inplace=True)
    data_vec = pd.concat([item_id_df, pd.DataFrame(data_vec)], axis=1)
    return data_vec

def quadratic_weighted_kappa(y_true, y_pred, labels = None):
    if labels is None:
        labels = np.unique(y_true)
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    print(conf_mat)
    num_ratings = len(labels)

    # Compute observed agreement
    observed_agreement = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

    # Compute expected agreement
    hist_true = np.sum(conf_mat, axis = 1)
    hist_pred = np.sum(conf_mat, axis = 0)
    expected_agreement = np.dot(hist_true, hist_pred) / np.sum(conf_mat) ** 2

    # Compute quadratic weights
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = (i - j) ** 2

    # Compute kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    quadratic_weighted_kappa = 1 - (
                np.sum(weights * conf_mat) / np.sum(weights * hist_true[:, None] * hist_pred[None, :]))

    return quadratic_weighted_kappa










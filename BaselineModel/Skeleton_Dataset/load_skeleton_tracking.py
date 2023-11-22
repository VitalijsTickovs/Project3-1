import math
import json
import numpy as np

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_ucla_data(path):
    train_1 = load_data(path + '/new_xyz_transform_skel_1.json') # error correction : added /
    train_2 = load_data(path + '/new_xyz_transform_skel_2.json') # error correction : added /
    train_data = {**train_1, **train_2}
    # train data
    for key in train_data.keys():
        train_data[key] = np.asarray(train_data[key])

    # test data
    test_data = load_data(path + '/new_xyz_transform_skel_3.json') # error correction : added /
    for key in test_data.keys():
        test_data[key] = np.asarray(test_data[key])
    # delete bad data
    del train_data['a02_s09_e04_v02']
    del test_data['a02_s09_e04_v03']
    # size of training and test data
    print("Size of training data: ", len(train_data))
    print("Size of test data: ", len(test_data))
    return train_data, test_data


def normalize_ucla(video):
    max_60 = np.amax(video, axis=0)
    min_60 = np.amin(video, axis=0)
    max_x = np.max([max_60[i] for i in range(0, 60, 3)])
    max_y = np.max([max_60[i] for i in range(1, 60, 3)])
    max_z = np.max([max_60[i] for i in range(2, 60, 3)])
    min_x = np.min([min_60[i] for i in range(0, 60, 3)])
    min_y = np.min([min_60[i] for i in range(1, 60, 3)])
    min_z = np.min([min_60[i] for i in range(2, 60, 3)])
    norm = np.zeros_like(video)
    for i in range(0, 60, 3):
        norm[:, i] = 2 * (video[:, i] - min_x) / (max_x - min_x) - 1
        norm[:, i + 1] = 2 * (video[:, i + 1] - min_y) / (max_y - min_y) - 1
        norm[:, i + 2] = 2 * (video[:, i + 2] - min_z) / (max_z - min_z) - 1
    return norm


def downsample_ucla(data):
    dsamp = dict()
    for key, val in data.items():
        if val.shape[0] > 50:
            new_val = np.zeros((50, 60))
            diff = math.floor(val.shape[0] / 50)
            idx = 0
            for i in range(0, val.shape[0], diff):
                new_val[idx, :] = val[i, :]
                idx += 1
                if idx >= 50:
                    break
            dsamp.update({key: new_val})
        else:
            dsamp.update({key: val})
    return dsamp


def get_feature_label(raw_data, dsamp_data, max_len=50):
    fea_xyz = []
    labels = []
    seq_len = []
    for key, val in raw_data.items():
        label = int(key[1:3])
        if label == 11:
            label = 7
        elif label == 12:
            label = 10
        label -= 1
        raw_len = val.shape[0]
        if raw_len > max_len:
            seq_len.append(max_len)
            fea_xyz.append(dsamp_data[key])
        else:
            seq_len.append(raw_len)
            pad_data = np.zeros((max_len, 60))
            pad_data[:raw_len, :] = dsamp_data[key]
            fea_xyz.append(pad_data)
        one_hot_label = np.zeros((10,))
        one_hot_label[label] = 1.
        labels.append(one_hot_label)
    return fea_xyz, labels, seq_len


def preprocess_ucla(path):
    # raw data
    train_data, test_data = load_ucla_data(path)
    # normalize
    for key in train_data.keys():
        train_data[key] = normalize_ucla(train_data[key])
    for key in test_data.keys():
        test_data[key] = normalize_ucla(test_data[key])
    # down sample
    dsamp_train = downsample_ucla(train_data)
    dsamp_test = downsample_ucla(test_data)
    # get features and labels pair
    tr_fea_xyz, tr_label, tr_seq_len = get_feature_label(train_data, dsamp_train, max_len=50)
    train_label = [np.argmax(tr_label[i]) for i in range(len(tr_label))]
    te_fea_xyz, te_label, te_seq_len = get_feature_label(test_data, dsamp_test, max_len=50)
    test_label = [np.argmax(te_label[i]) for i in range(len(te_label))]
    return dsamp_train, dsamp_test, tr_fea_xyz, tr_label, tr_seq_len, te_fea_xyz, te_label, te_seq_len


# dsamp_train, dsamp_test, tr_fea_xyz, tr_label, tr_seq_len, te_fea_xyz, te_label, te_seq_len = preprocess_ucla("Project3-1/BaselineModel/Skeleton_Dataset/ucla_data")
# print("dsamp_train: ", dsamp_train) # Squeleton sample train data vectorized under dictionary of 2D arrays
# print("dsamp_test: ", dsamp_test) # Squeleton sample test data vectorized under dictionary of 2D arrays

# print("tr_fea_xyz: ", tr_fea_xyz) # training feature vectors, len(tr_fea_xyz) = 1019 =  size of training data
# print("tr_label: ", tr_label) # training labels / target values, len(tr_label) = 1019 =  size of training data
# print("tr_seq_len: ", tr_seq_len) # sequence length of each individual feature xyz training vector, total vectors = 1019 = len(tr_seq_len) = len(tr_fea_xyz) = len(tr_label)

# print("te_fea_xyz: ", te_fea_xyz) # test feature vectors, len(te_fea_xyz) = 463 =  size of test data
# print("te_label: ", te_label) # test labels / target values, len(te_label) = 463 =  size of test data
# print("te_seq_len: ", te_seq_len) # sequence length of each individual feature xyz test vector, total vectors = 463 = len(te_seq_len) = len(te_fea_xyz) = len(te_label)
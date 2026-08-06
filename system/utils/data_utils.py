import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        if str(idx) == "all":
            all_x, all_y = [], []
            for file in os.listdir(train_data_dir):
                if file.endswith('.npz'):
                    data = np.load(os.path.join(train_data_dir, file), allow_pickle=True)['data'].tolist()
                    all_x.append(data['x'])
                    all_y.append(data['y'])
            return {'x': np.concatenate(all_x, axis=0), 'y': np.concatenate(all_y, axis=0)}

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        if str(idx) == "all":
            all_x, all_y = [], []
            for file in os.listdir(test_data_dir):
                if file.endswith('.npz'):
                    data = np.load(os.path.join(test_data_dir, file), allow_pickle=True)['data'].tolist()
                    all_x.append(data['x'])
                    all_y.append(data['y'])
            return {'x': np.concatenate(all_x, axis=0), 'y': np.concatenate(all_y, axis=0)}

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


# ========================================================================
# NEW: Global (Whole-Label) Test Data Loading
# ========================================================================
# These functions load from the test_global/ directory instead of test/.
# Each client's global test shard contains ALL 10 labels (balanced),
# unlike the skewed test/ data which only has 2 labels per client.
# This is used to measure generalization performance.
# ========================================================================

def read_data_global(dataset, idx):
    """Read whole-label test data from test_global/ directory."""
    test_data_dir = os.path.join('../dataset', dataset, 'test_global/')

    if str(idx) == "all":
        all_x, all_y = [], []
        for file in os.listdir(test_data_dir):
            if file.endswith('.npz'):
                data = np.load(os.path.join(test_data_dir, file), allow_pickle=True)['data'].tolist()
                all_x.append(data['x'])
                all_y.append(data['y'])
        return {'x': np.concatenate(all_x, axis=0), 'y': np.concatenate(all_y, axis=0)}

    test_file = test_data_dir + str(idx) + '.npz'
    with open(test_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()

    return test_data

def read_client_data_global(dataset, idx):
    """
    Load whole-label test shard for a specific client.
    Mirrors read_client_data() but reads from test_global/ instead of test/.
    Returns list of (x_tensor, y_tensor) tuples ready for DataLoader.
    """
    test_data = read_data_global(dataset, idx)
    X_test = torch.Tensor(test_data['x']).type(torch.float32)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


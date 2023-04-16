import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import controldiffeq

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)
    # X = tuple(TensorFloat(x) for x in X)
    # Y = TensorFloat(Y)
    data = torch.utils.data.TensorDataset(*X, torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


def get_dataloader_cde(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    # data = load_st_dataset(args.dataset)        # B, N, D
    # #normalize st data
    # data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    # #spilit dataset by days or by ratio
    # if args.test_ratio > 1:
    #     data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    # else:
    #     data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # #add time window
    # x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    # x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    # x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    # print('Train: ', x_tra.shape, y_tra.shape)
    # print('Val: ', x_val.shape, y_val.shape)
    # print('Test: ', x_test.shape, y_test.shape)
    #
    # Train: (10173, 12, 307, 1)(10173, 12, 307, 1)
    # Val: (3375, 12, 307, 1)(3375, 12, 307, 1)
    # Test: (3375, 12, 307, 1)(3375, 12, 307, 1)

    data_seq = np.load('../data/PEMS04/PEMS04.npz')['data']  # (sequence_length, num_of_vertices, num_of_features)
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, 0, 0,
                                    1, idx, 12,
                                    12)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        num_of_weeks = 0
        num_of_days = 0
        num_of_hours = 1
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2],
                             axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    mean_ = stats['_mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std_ = stats['_std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    print("mean:", mean_.shape)
    print("std:", std_.shape)

    train_x_final = train_x_norm[:, :, 0:1, :]
    val_x_final = val_x_norm[:, :, 0:1, :]
    test_x_final = test_x_norm[:, :, 0:1, :]

    train_target_final = np.expand_dims(train_target, axis=-1)
    val_target_final = np.expand_dims(val_target, axis=-1)
    test_target_final = np.expand_dims(test_target, axis=-1)

    x_tra = train_x_final.swapaxes(1, 2).swapaxes(1, 3)
    x_val = val_x_final.swapaxes(1, 2).swapaxes(1, 3)
    x_test = test_x_final.swapaxes(1, 2).swapaxes(1, 3)


    y_tra = train_target_final.swapaxes(1, 2)
    y_val = val_target_final.swapaxes(1, 2)
    y_test = test_target_final.swapaxes(1, 2)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    #
    # Train: (10173, 12, 307, 1)(10173, 12, 307, 1)
    # Val: (3375, 12, 307, 1)(3375, 12, 307, 1)
    # Test: (3375, 12, 307, 1)(3375, 12, 307, 1)

    # TODO: make argument for missing data
    if args.missing_test == True:
        generator = torch.Generator().manual_seed(56789)
        xs = np.concatenate([x_tra, x_val, x_test])
        for xi in xs:
            removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[:int(xs.shape[1] * args.missing_rate)].sort().values
            removed_points_node = torch.randperm(xs.shape[2], generator=generator)[:int(xs.shape[2] * args.missing_rate)].sort().values

            for seq in removed_points_seq:
                for node in removed_points_node:
                    xi[seq,node] = float('nan')
        x_tra = xs[:x_tra.shape[0],...]
        x_val = xs[x_tra.shape[0]:x_tra.shape[0]+x_val.shape[0],...]
        x_test = xs[-x_test.shape[0]:,...]
    ####
    # TODO: make argument for data category
    data_category = 'traffic'
    if data_category == 'traffic':
        times = torch.linspace(0, 11, 12)
    elif data_category == 'token':
        times = torch.linspace(0, 6, 7)
    else:
        raise ValueError
    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

    ####
    # train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, torch.Tensor(x_tra).transpose(1,2))
    # valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, torch.Tensor(x_val).transpose(1,2))
    # test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, torch.Tensor(x_test).transpose(1,2))
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1,2))
    # train_coeffs = tuple(coeff.transpose(1,2) for coeff in train_coeffs)
    # valid_coeffs = tuple(coeff.transpose(1,2) for coeff in valid_coeffs)
    # test_coeffs = tuple(coeff.transpose(1,2) for coeff in test_coeffs)
    ##############get dataloader######################
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
        # val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    # test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, mean_, std_, times

if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
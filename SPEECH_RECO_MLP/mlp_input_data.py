# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf


Batch_Size = 1000
phone = 374


def read_data(infile, name):
    mel_list = []
    id_list = []
    ids_list = []
    train_dir = os.path.join(infile, name)
    _files = os.listdir(train_dir)
    for i in _files:
        now_dir = os.path.join(train_dir, i)
        if now_dir.split('.')[-1] == 'mfcc':
            mel_ = np.loadtxt(now_dir)
            mel_.tolist()
            for x in mel_:
                for y in x:
                    y = (y-np.min(x))/(np.max(x)-np.max(x))
                mel_list.append(x)
        else:
            id_ = np.loadtxt(now_dir)
            id_.tolist()
            for x in id_:
                id_list.append(x)
                c = []
                for index in range(phone):
                    if index == int(x):
                        c.append(1)
                    else:
                        c.append(0)
                ids_list.append(c)
    mel_list = np.array(mel_list)
    mel_list = mel_list.reshape(-1, 39)
    ids_list = np.array(ids_list)
    ids_list = ids_list.reshape(-1, phone)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * Batch_Size

    return tf.train.shuffle_batch(
        [mel_list, ids_list], Batch_Size, capacity,
        min_after_dequeue)

import os
import cPickle as pickle
#import pickle
import time
import csv
import random
import itertools
import nltk
import math
import numpy as np
import json

import copy

import multiprocessing as mp

from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_20newsgroups

from collections import defaultdict

import random

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gc

class DataSet(object):
    def __init__(self, data, data_type):
        self.data = data 
        self.data_type = data_type
        self.num_examples = self.get_data_size()

    def get_data_size(self):
        return len(self.data) 

    def get_num_batches(self, batch_size):
        return int(math.ceil(self.num_examples/batch_size))

    def get_word_lists(self):
        words = {}

        for i in range(len(self.data)):
            for w in self.data[i][0]:
                if type(w) == list:
                    continue
                if w.find(" ") == -1:
                    w = w.lower()
                    if w in words:
                        words[w] += 1
                    else:
                        words[w] = 1
                else:
                    for w in w.split(" "):
                        w = w.lower()
                        if w in words:
                            words[w] += 1
                            words[w] = 1
        return words

    def get_by_idxs(self, idxs, batch_size, pad_to_full_batch):
        '''
        return data objects in a list based on the given idxs
        '''
        out = [copy.deepcopy(self.data[i]) for i in idxs]
        if pad_to_full_batch:
            while len(out) < batch_size:
                out.append(out[0])

        assert len(out) == batch_size

        return out

    def get_batches(self, batch_size, shuffle=True, front_heavy=False,
            pad_to_full_batch=False):
        # front_heavy: put long context first, so oov happens early
        # compute # of batches needed
        # pad_to_full_batch: pad until it's a full batch
        if pad_to_full_batch:
            num_batches = int(math.ceil(float(self.num_examples)/batch_size))
        else:
            num_batches = self.num_examples / batch_size

        idx = range(self.num_examples)
        if front_heavy:
            idx = sorted(idx, key=lambda i:
                    len(self.data[i]['summary']) * len(self.data[i]['answer1']),
                    reverse=True)
            #print(len(self.data[idx[0]]['summary']))
        elif shuffle:
            random.shuffle(idx)

        for i in range(num_batches):
            idx_slice = idx[i * batch_size : (i+1) * batch_size]
            yield self.get_by_idxs(idx_slice, batch_size, pad_to_full_batch)

def create_twenty_newsgroup_data(config):
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers'))
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers'))

    train_processed = []
    test_processed = []

    for i, sent in enumerate(train.data):
        sent = nltk.word_tokenize(sent.lower())
        train_processed.append((sent, train.target[i]))

    for i, sent in enumerate(test.data):
        sent = nltk.word_tokenize(sent.lower())
        test_processed.append((sent, test.target[i]))

    random.shuffle(train_processed)

    dev_split = train_processed[:int(len(train_processed) * 0.1)]
    train_split = train_processed[int(len(train_processed) * 0.1):]

    return DataSet(train_split, "train"), DataSet(dev_split, "dev"), \
        DataSet(test_processed, "test")

def create_imdb_data(config):
    def build_dataset(root):
        data = []
        labels = []
        for filename in os.listdir(os.path.join(root, 'pos')):
            with open(os.path.join(root, 'pos', filename), 'r') as in_f:
                a = in_f.read()
            a = a.encode("utf-8")
            data.append(a)
            labels.append(1)
        for filename in os.listdir(os.path.join(root, 'neg')):
            with open(os.path.join(root, 'neg', filename), 'r') as in_f:
                a = in_f.read()
            a = a.encode("utf-8")
            data.append(a)
            labels.append(0)

        return data, labels

    train_data, train_target = build_dataset(os.path.join(
        config.data_root, 'train'))
    test_data, test_target = build_dataset(os.path.join(
        config.data_root, 'test'))

    train_processed = []
    test_processed = []

    for i, sent in enumerate(train_data):
        try:
            sent = nltk.word_tokenize(sent.lower())
            train_processed.append((sent, train_target[i]))
        except UnicodeDecodeError:
            pass

    for i, sent in enumerate(test_data):
        try:
            sent = nltk.word_tokenize(sent.lower())
            test_processed.append((sent, test_target[i]))
        except UnicodeDecodeError:
            pass

    random.shuffle(train_processed)

    dev_split = train_processed[:int(len(train_processed) * 0.1)]
    train_split = train_processed[int(len(train_processed) * 0.1):]

    return DataSet(train_split, "train"), DataSet(dev_split, "dev"), \
        DataSet(test_processed, "test")



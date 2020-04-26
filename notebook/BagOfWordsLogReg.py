
from typing import *

import nltk
import random
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.linear_model import LogisticRegression

class BoWLR(object):
    def __init__(self,
            data: List[str],
            labels: List[int],
            rare_word_threshold : int = 5,
            dev_percentage : float = 0.1):

        self._stemmer = PorterStemmer()
        self._stopwords = set(stopwords.words('english'))
        self._rare_word_threshold = rare_word_threshold
        self._dev_percentage = dev_percentage

        self._build_vocab(data)
        self._build_train_dev(data, labels)

        self._clf = LogisticRegression(verbose=True)
    
    def _build_vocab(self, data):
        freq = {}

        for sent in data:
            for w in word_tokenize(sent.lower()):
                if w in self._stopwords:
                    continue
                w = self._stemmer.stem(w)
                if w in freq:
                    freq[w] += 1
                else:
                    freq[w] = 1

        self._word2id = {}
        self._id2word = []
        for w, count in freq.items():
            if count < self._rare_word_threshold:
                continue
            self._word2id[w] = len(self._id2word)
            self._id2word.append(w)

        self._vocab_size = len(self._word2id)

        print("Done building vocab! Vocab size: %d" % len(self._word2id))

    def _build_train_dev(self, data, labels):
        combined_data = list(zip(data, labels))
        random.shuffle(combined_data)

        data = [d for (d, l) in combined_data]
        labels = [l for (d, l) in combined_data]

        dev_percentage = self._dev_percentage

        dev_raw = data[:int(len(data) * dev_percentage)]
        self._dev_X = self._preprocess_X(dev_raw)
        self._dev_y = labels[:int(len(data) * dev_percentage)]

        train_raw = data[int(len(data) * dev_percentage):]
        self._train_X = self._preprocess_X(train_raw)
        self._train_y = labels[int(len(data) * dev_percentage):]


        print("Done building train dev!")

    def _preprocess_X(self, data):
        data_X = np.zeros((len(data), self._vocab_size), dtype=np.int32)
        for i, sent in enumerate(data):
            for w in word_tokenize(sent):
                w = self._stemmer.stem(w)

                if w in self._word2id:
                    data_X[i, self._word2id[w]] = 1
        return data_X


    def fit(self):
        print("Training...")
        self._clf.fit(self._train_X, self._train_y)
        print("Training done!")
        print("Testing on dev...")
        score = self._clf.score(self._dev_X, self._dev_y)
        print("Done! Scores below:")
        print(score)

    def test(self, data, labels, preprocessed=False):
        if not preprocessed:
            data = self._preprocess_X(data)

        print(self._clf.score(data, labels))

    def classify(self, data):
        preprocessed_X = self._preprocess_X(data)
        return self._clf.predict(preprocessed_X)

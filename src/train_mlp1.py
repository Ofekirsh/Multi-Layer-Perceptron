import mlp1 as mlp
import random
from collections import Counter
import numpy as np
from xor_data import data as xor_data


def read_data(fname):
    with open(fname, encoding="utf8") as f:
        data = []
        for line in f:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


def text_to_unigrams(text):
    return ["%s" % (c1) for c1 in text]

def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def feats_to_vec(features):
    counts_array = np.zeros(len(vocab_list), dtype=int)
    for i, string in enumerate(vocab_list):
        counts = features.count(string)
        counts_array[i] = counts
    return counts_array


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        features_vec = feats_to_vec(features)
        prediction = mlp.predict(features_vec, params)
        if prediction == L2I[label]:
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)
            y = L2I[label]
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            for param, grad in zip(params, grads):
                param -= learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # If you want to change according to unigram then change the following two lines to the following commands:
    # train_data = [(l, text_to_unigrams(t)) for l, t in read_data("train")]
    # dev_data = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]

    train_data = [(l, text_to_bigrams(t)) for l, t in read_data("../data/train")]
    dev_data = [(l, text_to_bigrams(t)) for l, t in read_data("../data/dev")]

    fc = Counter()
    for l, feats in train_data:
        fc.update(feats)

    vocab_size = 600
    vocab = set([x for x, c in fc.most_common(vocab_size)])
    vocab_list = list(sorted(vocab))

    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}
    I2L = {i: l for l, i in L2I.items()}

    in_dim = len(vocab_list)
    hid_dim = 100  # hyperparameter
    out_dim = len(L2I)

    params = mlp.create_classifier(in_dim, hid_dim, out_dim)

    num_iterations = 20
    learning_rate = 0.01
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # the code for xor data is in the file xor_data.py, for run this code only write the command:
    # python xor_data.py

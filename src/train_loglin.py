import loglinear as ll
import random
from collections import Counter
import numpy as np


def read_data(fname):
    with open(fname, encoding="utf8") as f:
        data = []
        for line in f:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data

def read_blind_test_data(fname):
    with open(fname, encoding="utf8") as f:
        data = []
        for line in f:
            text = line.strip().lower()
            data.append(text)
    return data

def text_to_unigrams(text):
    return list(text)

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
        prediction = ll.predict(features_vec, params)
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
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            for param, grad in zip(params, grads):
                param -= learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


#for prediction file
def predict_test_data(test_data, params, text_to_features):
    predictions = []
    for text in test_data:
        features = text_to_features(text)
        features_vec = feats_to_vec(features)
        prediction = ll.predict(features_vec, params)
        predictions.append(prediction)
    return predictions

if __name__ == '__main__':
    feature_extractor = input("Enter 'unigram' for unigrams or 'bigram' for bigrams: ").strip().lower()
    if feature_extractor == 'bigram':
        text_to_features = text_to_bigrams
    else:
        text_to_features = text_to_unigrams

    train_data = [(l, text_to_features(t)) for l, t in read_data("../data/train")]
    dev_data = [(l, text_to_features(t)) for l, t in read_data("../data/dev")]
    test_data = read_blind_test_data("test")

    fc = Counter()
    for l, feats in train_data:
        fc.update(feats)

    vocab_size = 600
    vocab = set([x for x, c in fc.most_common(vocab_size)])
    vocab_list = list(sorted(vocab))

    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}
    I2L = {i: l for l, i in L2I.items()}

    params = ll.create_classifier(len(vocab_list), len(L2I))
    num_iterations = 30
    learning_rate = 0.02
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # for prediction file, if you want to print a new file remove the - """
    """
    predictions = predict_test_data(test_data, trained_params, text_to_features)
    len_pred = len(predictions)
    with open("test.pred", "w", encoding="utf8") as f:
        for i, pred in enumerate(predictions):
            if i == len_pred - 1:
                f.write(f"{I2L[pred]}")
            else:
                f.write(f"{I2L[pred]}\n")
    """
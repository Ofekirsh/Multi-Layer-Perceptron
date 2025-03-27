import mlp1 as mlp  # Importing functions from mlp1.py
import random
import numpy as np


data = [(1, [0, 0]),
        (0, [0, 1]),
        (0, [1, 0]),
        (1, [1, 1])]

STUDENT = {'name': 'YOUR NAME', 'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    return np.array(features)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        features_vec = feats_to_vec(features)
        prediction = mlp.predict(features_vec, params)
        if prediction == int(label):
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier_xor_data(train_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)
            y = int(label)
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss
            for param, grad in zip(params, grads):
                param -= learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        print(f"Iteration {I}: Loss = {train_loss}, Training Accuracy = {train_accuracy}")
        if train_accuracy == 1.0:
            print(f"Solved XOR in {I + 1} iterations")
            return I + 1  # Return the number of iterations taken to solve XOR
    return num_iterations

if __name__ == '__main__':
    train_data = data

    input_size = 2  # XOR inputs are 2-dimensional
    hid_dim = 100
    output_size = 2  # Two possible outputs (0 or 1)

    params = mlp.create_classifier(input_size, hid_dim, output_size)
    num_iterations = 10000
    learning_rate = 0.01
    iterations_taken = train_classifier_xor_data(train_data, num_iterations, learning_rate, params)


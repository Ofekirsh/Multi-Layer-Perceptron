import numpy as np
from grad_check import gradient_check


def _loss_and_w_grad(w):
    global b, U, b_tag
    params = [w, b, U, b_tag]
    loss, grads = loss_and_gradients([1, 2, 3], 0, params)
    return loss, grads[0]


def _loss_and_b_grad(b):
    global W, U, b_tag
    params = [W, b, U, b_tag]
    loss, grads = loss_and_gradients([1, 2, 3], 0, params)
    return loss, grads[1]


def _loss_and_u_grad(u):
    global W, b, b_tag
    params = [W, b, u, b_tag]
    loss, grads = loss_and_gradients([1, 2, 3], 0, params)
    return loss, grads[2]


def _loss_and_b_tag_grad(b_tag):
    global W, b, U
    params = [W, b, U, b_tag]
    loss, grads = loss_and_gradients([1, 2, 3], 0, params)
    return loss, grads[3]


def classifier_output(x, params):
    """
    Compute the output probabilities of the classifier using the softmax function.
    """
    W, b, U, b_tag = params

    # Compute the linear transformation for the hidden layer
    z1 = np.dot(x, W) + b  # Shape: (hidden_dim,)

    # Apply the tanh activation function
    a1 = np.tanh(z1)  # Shape: (hidden_dim,)

    # Compute the linear transformation for the output layer
    z2 = np.dot(a1, U) + b_tag  # Shape: (output_dim,)

    # Apply the softmax function to get the probabilities
    exp_z2 = np.exp(z2 - np.max(z2))
    probs = exp_z2 / exp_z2.sum()  # Shape: (output_dim,)

    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """

    W, b, U, b_tag = params

    # Forward pass
    z1 = np.dot(x, W) + b
    a1 = np.tanh(z1)
    z2 = np.dot(a1, U) + b_tag
    exp_z2 = np.exp(z2 - np.max(z2))  # for numerical stability
    probs = exp_z2 / exp_z2.sum()

    # Compute loss
    loss = -np.log(probs[y])

    # Backward pass
    """
    x^0 - features
    x^1 = f(Wx^0 + b) , z_1 = Wx^0 + b
    x^2 = f(Ux^1 + b_tag), z_2 = Ux^1 + b_tag
    y_hat = prediction
    dL/dz^k = dL/dx^k * dx^k/dz^k = 
     
    gU = dL/dU = dL/dz^2 * dz^2/dU = (y_hat - h) * df(z_2)/dU = (y_hat - h) * x^1.transpose 
    gb_tag = dL/db_tag = dL/dz^2 * dz^2/db_tag = (y_hat - h) * df(z_2)/db_tag = (y_hat - h) * 1
    gW = dL/dW = dL/dz^1 * dz^1/dW =  (U.transpose * (y_hat - y)) * df(z_1)/dW = (U.transpose * (y_hat - y)) * x^0.transpose
    gb = dL/db = dL/dz^1 * dz^1/db = (U.transpose * (y_hat - y)) * df(z_1)/db = (U.transpose * (y_hat - y)) * 1
    """
    y_one_hot = np.zeros_like(probs)
    y_one_hot[y] = 1
    g_z2 = probs - y_one_hot
    gU = np.outer(a1, g_z2)
    gb_tag = g_z2

    g_a1 = np.dot(U, g_z2)
    g_z1 = g_a1 * (1 - np.tanh(z1) ** 2)
    gW = np.outer(x, g_z1)
    gb = g_z1

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    num_w = (6 ** 0.5) / ((in_dim + hid_dim) ** 0.5)
    num_u = (6 ** 0.5) / ((hid_dim + out_dim) ** 0.5)

    W = np.random.uniform(-num_w, num_w, size=(in_dim, hid_dim))
    b = np.random.uniform(-num_w, num_w, size=hid_dim)
    U = np.random.uniform(-num_u, num_u, size=(hid_dim, out_dim))
    b_tag = np.random.uniform(-num_u, num_u, size=out_dim)

    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    # Create classifier
    W, b, U, b_tag = create_classifier(3, 4, 2)

    # Perform gradient check
    for _ in range(10):
        W = np.random.randn(*W.shape)
        b = np.random.randn(*b.shape)
        U = np.random.randn(*U.shape)
        b_tag = np.random.randn(*b_tag.shape)

        gradient_check(_loss_and_w_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_u_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)

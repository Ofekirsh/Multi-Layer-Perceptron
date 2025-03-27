import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)
def classifier_output(x, params):
    # YOUR CODE HERE.
    z_layers = []
    h_layers = []
    h = x

    for i in range(0, len(params), 2):
        w, b = params[i], params[i + 1]
        z = np.dot(h, w) + b
        h = np.tanh(z)
        z_layers.append(z)
        h_layers.append(h)

    z_layers.pop()
    h_layers.pop()
    probs = softmax(z)

    return probs, z_layers, h_layers

def predict(x, params):
    probs, _, _ = classifier_output(x, params)
    return np.argmax(probs)


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    pred_vec, z_layers, h_layers = classifier_output(x, params)
    loss = -np.log(pred_vec[y])

    y_hot_vector = np.zeros_like(pred_vec)
    y_hot_vector[y] = 1

    grad_output = pred_vec - y_hot_vector
    grads = []
    grads.insert(0, grad_output)

    Ws = params[0::2]

    for i in range(len(Ws) - 1):
        dz_dw = h_layers[-(i + 1)].T
        grad_w = np.outer(dz_dw, grad_output)
        grads.insert(0, grad_w)

        U = Ws[-(i + 1)]
        grad_output = np.dot(grad_output, U.T) * (1 - np.tanh(z_layers[-(i + 1)]) ** 2)
        grads.insert(0, grad_output)

    grad_first_w = np.outer(x, grad_output)
    grads.insert(0, grad_first_w)

    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []

    for dim1, dim2 in zip(dims, dims[1:]):
        epsilon_w = np.sqrt(6) / np.sqrt(dim1 + dim2)
        W = np.random.uniform(-epsilon_w, epsilon_w, (dim1, dim2))
        epsilon_b = np.sqrt(6) / np.sqrt(dim2)
        b = np.random.uniform(-epsilon_b, epsilon_b, dim2)
        params.extend([W, b])

    return params



if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag, V, b_tag_tag = create_classifier([2, 2, 2, 2])


    def _loss_and_W_grad(W):
        global b, U, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W, U, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W, b, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W, U, b, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[3]


    def _loss_and_V_grad(V):
        global W, U, b, b_tag, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[4]


    def _loss_and_b_tag_tag_grad(b_tag_tag):
        global W, U, b, V, b_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[5]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        U = np.random.randn(U.shape[0], U.shape[1])
        V = np.random.randn(V.shape[0], V.shape[1])
        b = np.random.randn(b.shape[0])
        b_tag = np.random.randn(b_tag.shape[0])
        b_tag_tag = np.random.randn(b_tag_tag.shape[0])

        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_V_grad, V)
        gradient_check(_loss_and_b_tag_tag_grad, b_tag_tag)
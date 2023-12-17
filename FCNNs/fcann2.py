import numpy as np
from data import *

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)


def fcann2_train(X, Y_, param_niter = 100000, param_delta = 0.05, param_lambda = 1e-3, param_hidden_layer_size = 5):
    C = int(max(Y_) + 1)
    N = X.shape[0]
    W1 = np.random.randn(X.shape[1], param_hidden_layer_size) # D x Neurons
    # print('W1 shape: ',W1.shape)
    b1 = np.zeros((1, param_hidden_layer_size)) # 1 x Neurons
    # print('b1 shape: ',b1.shape)
    W2 = np.random.randn(param_hidden_layer_size, C) # Neurons x C
    # print('W2 shape: ',W2.shape)
    b2 = np.zeros((1, C)) # 1 x C
    # print('b2 shape: ',b2.shape)

    for i in range(param_niter):
        # forward pass
        scores1 = np.dot(X, W1) + b1
        # print('Scores1 shape: ',scores1.shape)
        hidden_layer = np.maximum(0, scores1)
        # print('Hidden layer shape: ',hidden_layer.shape)
        scores2 = np.dot(hidden_layer, W2) + b2
        # print('Scores2 shape: ',scores2.shape)
        probs = softmax(scores2)
        # print('Probs shape: ',probs.shape)

        one_hot = class_to_onehot(Y_)
        # print('One hot shape: ',one_hot.shape)

        # loss
        data_loss = -np.sum(one_hot * np.log(probs)) / N
        reg_loss = 0.5 * param_lambda * np.sum(W1 * W1) + 0.5 * param_lambda * np.sum(W2 * W2)
        loss = data_loss + reg_loss

        # trace
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # backpropagation
        dL_dscores2 = probs - one_hot
        dL_dW2 = np.dot(hidden_layer.T, dL_dscores2) / N + param_lambda * W2
        dL_db2 = np.sum(dL_dscores2, axis=0, keepdims=True) / N
        dL_dhidden_layer = np.dot(dL_dscores2, W2.T)
        dL_dscores1 = dL_dhidden_layer
        dL_dscores1[scores1 <= 0] = 0
        dL_dW1 = np.dot(X.T, dL_dscores1) / N + param_lambda * W1
        dL_db1 = np.sum(dL_dscores1, axis=0, keepdims=True) / N

        # update parameters
        W1 -= param_delta * dL_dW1
        b1 -= param_delta * dL_db1
        W2 -= param_delta * dL_dW2
        b2 -= param_delta * dL_db2

    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    scores1 = np.dot(X, W1) + b1
    hidden_layer = np.maximum(0, scores1)
    scores2 = np.dot(hidden_layer, W2) + b2
    probs = softmax(scores2)
    return probs

def decfun(W1, b1, W2, b2):
    def classify(X):
      return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # create the inputs and labels
    X, Y_ = sample_gmm_2d(6, 2, 10)
    # Shape of X: (N, D) N = number of samples = ncomponents x nsamples,
    # D = dimensionality of each sample = 2 for 2d data
    # Shape of Y_: (N,)

    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, param_hidden_layer_size=5)

    # print('Y_: ', Y_)
    # evaluate the model on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

    # graph the decision surface
    decfun = decfun(W1, b1, W2, b2)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()
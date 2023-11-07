import numpy as np

from data import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binlogreg_train(X, Y_, param_niter = 5000):
    '''
    Arguments
    X:  data, np.array NxD
    Y_: class indices, np.array Nx1

    Return values
    w, b: parameters of binary logistic regression
    '''

    w = np.random.randn(X.shape[1], 1)
    b = 0
    
    param_delta = 0.1
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = sigmoid(scores)
        loss = -np.mean(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))
        # trace
        # if i % 10 == 0:
        #     print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores
        dL_dscores = probs - Y_
        # derivative of the loss function with respect to parameters
        dL_dw = np.mean(X * dL_dscores, axis=0)
        dL_db = np.mean(dL_dscores)
        # update parameters
        w -= param_delta * dL_dw[:, np.newaxis]
        b -= param_delta * dL_db

    return w, b

def binlogreg_classify(X, w, b):
    '''
    Arguments
    X:    data, np.array NxD
    w, b: logistic regression parameters
    Return values
    probs: a posteriori probabilities for c1, dimensions Nx1
    '''

    scores = np.dot(X, w) + b
    probs = sigmoid(scores)
    return probs

def binlogreg_decfun(w,b):
    def classify(X):
      return binlogreg_classify(X, w, b)
    return classify

def onehot_encode(Y_):
    '''
    Arguments
    Y_: class indices, np.array Nx1

    Return values
    Y:  one-hot encoded classes, np.array NxC
    '''

    Y_ = np.copy(Y_)
    Y_ = Y_.flatten()
    N = int(Y_.shape[0])
    C = int(max(Y_) + 1)
    Y = np.zeros((N, C))
    Y_ = Y_.astype(int)
    Y[np.arange(N), Y_] = 1
    return Y

def logreg_train(X, Y_, param_niter = 5000, param_delta = 0.1):
    # exponentiated classification scores
    # take into account how softmax is calculated
    # in section 4.1 of the text book
    # (Deep Learning, Goodfellow et al)!
    # define W and b for multinomial logistic regression
    # and initialize them randomly
    C = int(max(Y_) + 1)
    W = np.random.randn(X.shape[1], C)
    b = np.zeros((1, C))

    for i in range(param_niter):
        scores = np.dot(X, W) + b
        expscores = np.exp(scores)
        
        # divisor of the softmax function
        sumexp = np.sum(expscores, axis=1, keepdims=True)

        # logarithms of aposteriori probabilities 
        probs = expscores / sumexp
        logprobs = np.log(probs)

        # loss
        N = X.shape[0]
        onehot = onehot_encode(Y_)
        loss = -np.sum(onehot * logprobs) / N
        print('Loss Shape: ',loss.shape)
        
        # diagnostic trace
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))


        hot_encoded = onehot_encode(Y_)

        # derivative of the loss with respect to scores
        dL_ds = probs - hot_encoded

        # derivative of the loss with respect to the parameters
        grad_W = np.dot(X.T, dL_ds) / N
        grad_b = np.mean(dL_ds, axis=0)

        # modification of the parameters
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    
    return W, b



if __name__=="__main__":
    # np.random.seed(100)

    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 50)

    # train the model
    w,b = binlogreg_train(X, Y_)
    # w,b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)

    Y = probs > 0.5
    Y = Y.astype(int)

    # report performance
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    probs = probs.flatten()
    AP = eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)

    graph_data(X, Y_, Y)
    plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchviz import make_dot
import os

from data import *


class PTDeep(nn.Module):
    def __init__(self, config, activation=torch.relu):
        super(PTDeep, self).__init__()
        self.layers = len(config) - 1
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(config[i], config[i+1])) for i in range(self.layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(1, config[i+1])) for i in range(self.layers)])
        self.activation = activation
        self.flag = True
    def forward(self, X):
        self.Y_ = X
        # if self.flag:
        #     for i in range(self.layers):
        #         print('weight gradients: ',self.weights[i].requires_grad)
        #     for i in range(self.layers):
        #         print('bias gradients: ',self.biases[i].requires_grad)
            
        for i in range(self.layers):
            # print(str(self.Y_.shape) + ' X ' + str(self.weights[i].shape) + ' + ' + str(self.biases[i].shape))
            self.Y_ = torch.mm(self.Y_, self.weights[i]) + self.biases[i]
            # if self.flag:
            #     print('Y gradients: ', self.Y_.requires_grad)
            #     self.flag = False

            if i != self.layers - 1:
                self.Y_ = self.activation(self.Y_)
            else:
                max_values, indices = torch.max(self.Y_, dim=1)
                max_values = max_values.view(-1, 1)
                self.Y_ = self.Y_ - max_values
                self.prob = torch.softmax(self.Y_, dim=1)
                # print('Probabilities: ', self.prob)
    

    def get_loss(self, X, Yoh_, param_lambda=1e-3):
        # Add regularization in a way that you form the loss as a sum 
        # of cross entropy and the L2 norm of the vectorized weight 
        # matrix multiplied with a hyperparameter param_lambda. 
        vectorized_weights = torch.cat([self.weights[i].view(-1) for i in range(self.layers)])
        L2 = torch.norm(vectorized_weights, p=2)
        self.loss = (torch.sum(-torch.log(self.prob + 1e-10) * Yoh_) / X.shape[0]) + (param_lambda * L2)

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1e-3):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    # optimizer = optim.Adam(model.parameters(), lr=param_delta)

    # training loop
    for i in range(param_niter):
        # forward pass
        model.forward(X)
        # loss
        model.get_loss(X, Yoh_, param_lambda=param_lambda)
        # backward pass
        model.loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name} - Gradient: {param.grad}")
        # break
        # parameter update
        optimizer.step()
        # gradient reset
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f'Iteration: {i}, loss: {model.loss}')

def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
    """
    X_tensor = torch.Tensor(X)
    model.forward(X_tensor)
    return torch.Tensor.numpy(model.prob.detach())

def decfun(model):
    def classify(X):
      return np.argmax(eval(model, X), axis=1)
    return classify


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__== "__main__":

    # initialize random number generator
    np.random.seed(100)

    # define input data X and labels Yoh_
    # X, Y_ = sample_gauss_2d(3, 100)
    X, Y_ = sample_gmm_2d(6, 2, 80)
    Yoh_ = class_to_onehot(Y_)
    

    ptlr = PTDeep([2, 10, 10, 2], torch.relu)

    # learn the parameters (X and Yoh_ have to be of type torch.Tensor):
    train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, 0.01, param_lambda=1e-4)

    # Get the directory of the current script to save the computation graph
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # generate computation graph
    make_dot(ptlr.prob, params=ptlr.state_dict()).render("pt_deep", directory=script_directory,
                                                          format="png", cleanup=True)

    # get probabilites on training data
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)
    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))
    print('Total number of parameters: ',count_params(ptlr))

    # visualize the results, decicion surface
    # graph the decision surface
    decfun = decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0)
    graph_data(X, Y_, Y)
    plt.show()

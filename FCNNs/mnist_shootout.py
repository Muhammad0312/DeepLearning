import torch
import torchvision
from pt_deep import *
import os
import random

# Get the directory of the current script to save the computation graph
script_directory = os.path.dirname(os.path.abspath(__file__))

dataset_root = 'FCNNs/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

# Visualize random images from the training set
# random_integers = [random.randint(0, 30000) for _ in range(100)]
# for i in random_integers:
#     plt.imshow(x_train[i].numpy(), cmap='gray')
#     plt.show()


N=x_train.shape[0]
D=x_train.shape[1]*x_train.shape[2]
C=y_train.max().add_(1).item()

y_train_oh = class_to_onehot(y_train)

ptlr = PTDeep([D, 100, C])

train(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 10000, 0.1)
# generate computation graph
make_dot(ptlr.prob, params=ptlr.state_dict()).render("MNISTComputationGraph",
                                                      directory=script_directory, format="png", cleanup=True)

Weights = ptlr.weights
for i, w in enumerate(Weights):
    for i in range(w.size(1)):
        weight = w[:, i].detach().view(28, 28).numpy()
        plt.imshow(weight, cmap='gray')
        plt.title('Weights for class {}'.format(i))
        plt.show()

# get probabilites on training data
probs = eval(ptlr, x_train.view(N, D))
Y = np.argmax(probs, axis=1)
# print out the performance metric (precision and recall per class)
accuracy, recall, precision = eval_perf_multi(Y, y_train)
print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))
print('Total number of parameters: ', count_params(ptlr))
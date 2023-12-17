import torch
import torchvision
from pt_deep import *
import os
from sklearn.svm import SVC


def train_mb(model, X, Y, iterations, lr, param_lambda, mini_batches=100, optimizer=optim.SGD):
    N = X.shape[0]
    stored_loss = []
    
    for i in range(iterations):
        shufled_list = torch.randperm(N)
        batch_size = int(N/mini_batches)
        for j in range(mini_batches):
            idx = shufled_list[j*batch_size:(j+1)*batch_size]
            X_mb = X[idx]
            Y_mb = Y[idx]
            train(model, X_mb, Y_mb, 1, lr, param_lambda, optimizer)
            stored_loss.append(model.loss.detach().numpy())
        if i % 100 == 0:
            print(f'Iteration: {i}, loss:{model.loss}')
    return stored_loss

def train_early_stop(model, X, Y, iteration, lr, param_lambda, X_validation, Y_validation):
    fl = True
    for i in range(iteration):
        train(model, X, Y, 1, lr, param_lambda)
        probs = eval(model, X_validation.view(X_validation.shape[0], X_validation.shape[1]*X_validation.shape[2]))
        Y_true = np.argmax(probs, axis=1)
        accuracy, recall, precision = eval_perf_multi(Y_true, Y_validation)
        if i % 100 == 0:
            print(f'Iteration: {i}, loss:{model.loss:.3f}, accuracy: {accuracy:.3f}')
        if model.loss < 0.91 and fl:
            weights = model.weights
            biases = model.biases
            loss = model.loss
            fl = False

    return weights, biases, loss

# Make a function to train and evaluate linear SVM classifier using the module
# sklearn.svm.SVC by using one v one SVM variant for multiclasss classification
def train_svm(X, Y):
    print('Training Linear SVM')
    svm = SVC(kernel='linear', decision_function_shape='ovo')
    svm.fit(X, Y)
    Y_predicted = svm.predict(X)
    accuracy, recall, precision = eval_perf_multi(Y_predicted, Y)
    print('Linear SVM')
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

# Make a function to train and evaluate kernel SVM classifier using the module
# sklearn.svm.SVC by using one v one SVM variant for multiclasss classification
def train_kernel_svm(X, Y):
    print('Training Kernel SVM')
    svm = SVC(kernel='rbf', decision_function_shape='ovo')
    svm.fit(X, Y)
    Y_predicted = svm.predict(X)
    accuracy, recall, precision = eval_perf_multi(Y_predicted, Y)
    print('Kernel SVM')
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

if __name__ == '__main__':
    # Get the directory of the current script to save the computation graph
    script_directory = os.path.dirname(os.path.abspath(__file__))

    dataset_root = 'FCNNs/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    # # Visualize random images from the training set
    # random_integers = [random.randint(0, 30000) for _ in range(200)]
    # for i in random_integers:
    #     plt.imshow(x_train[i].numpy(), cmap='gray')
    #     plt.show()


    N=x_train.shape[0]
    D=x_train.shape[1]*x_train.shape[2]
    C=y_train.max().add_(1).item()

    y_train_oh = class_to_onehot(y_train)

    N_validation = N//5
    shufled_data_list = torch.randperm(N)

    x_validation = x_train[shufled_data_list[:N_validation]]
    y_validation = y_train[shufled_data_list[:N_validation]]

    ptlr = PTDeep([D, C])

    '''
    Training functions
    '''
    # # Train with full batch
    # train(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 10000, 0.2)

    # # Train with early stopping
    # weights, biases, loss = train_early_stop(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 6000, 0.2, 0.1, x_validation, y_validation)


    # Train with mini-batches
    stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 2000, 0.1, 0.1, 100, optimizer=optim.SGD)
    plt.plot(stored_loss)
    plt.show()

    # # Train using linear SVM
    # train_svm(x_train.view(N, D).numpy(), y_train.numpy())

    # # Train using kernel SVM
    # train_kernel_svm(x_train.view(N, D).numpy(), y_train.numpy())


    # generate computation graph
    make_dot(ptlr.prob, params=ptlr.state_dict()).render("MNISTComputationGraph",
                                                        directory=script_directory, format="png", cleanup=True)

    Weights = ptlr.weights
    for i, w in enumerate(Weights):
        for i in range(w.size(1)):
            weight = w[:, i].detach().view(28, 28).numpy()
            weight = (((weight - weight.min()) / (weight.max() - weight.min())) * 255.0).astype(np.uint8)
            plt.imshow(weight, cmap='gray')
            plt.title('Weights for class {}'.format(i))
            plt.show()

    # torch.save(ptlr.state_dict(), 'FCNNs/saved_weights/model_weights.pth')


    print('Total number of parameters: ', count_params(ptlr))

    # # Print evaluation metrics for the training set
    # get probabilites on training data
    print('Training set Metrics')
    probs = eval(ptlr, x_train.view(N, D))
    Y = np.argmax(probs, axis=1)
    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, y_train)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))



    # # Print evaluation metrics for the test set
    N_test = x_test.shape[0]
    D_test = x_test.shape[1]*x_test.shape[2]
    probs = eval(ptlr, x_test.view(N_test, D_test))

    # get probabilites on training data
    print('Test set Metrics')
    probs = eval(ptlr, x_test.view(N_test, D_test))
    Y = np.argmax(probs, axis=1)
    # print out the performance metric (precision and recall per class)
    accuracy, recall, precision = eval_perf_multi(Y, y_test)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))


    # # Print evaluation metrics for the validation set
    # print('Validation set Metrics')
    # N_validation = x_validation.shape[0]
    # D_validation = x_validation.shape[1]*x_validation.shape[2]
    # probs = eval(ptlr, x_validation.view(N_validation, D_validation))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_validation)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))


    # # Print evaluation metrics for the early stopping
    # ptlr.weights = weights
    # ptlr.biases = biases

    # print('Early stopping Metrics')
    # probs = eval(ptlr, x_test.view(N_test, D_test))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, recall, precision = eval_perf_multi(Y, y_test)
    # print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

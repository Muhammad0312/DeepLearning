import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import pdb
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class PTCIFAR(nn.Module):
    def __init__(self):
        super(PTCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding='same', bias=True)
        # output feature map size is 32x32x16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        #  out is 15x15x16
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding='same', bias=True)
        self.relu2 = nn.ReLU()
        # out is 15x15x32
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten3 = nn.Flatten()
        # out is 32*7*7
        self.fc3 = nn.Linear(32*7*7, 256, bias=True)
        # out is 256
        self.relu3 = nn.ReLU()
        # out is 256
        self.fc4 = nn.Linear(256, 128, bias=True)
        # out is 128
        self.relu4 = nn.ReLU()
        # out is 128
        self.logits = nn.Linear(128, 10, bias=True)
        # out is 10

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.logits.reset_parameters()

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.fc3(self.flatten3(x)))
        x = self.relu4(self.fc4(x))
        x = self.logits(x)
        return x

def train(model, train_dataloader, validation_dataloader, epochs, optimizer, loss_fn, writer):
    print('Starting Training')
    # evaluate the model on the training and validation set and log the results
    initial_accuracy_train = evaluate(model, train_dataloader)[0].item()
    initial_accuracy_validation = evaluate(model, validation_dataloader)[0].item()
    print('Initial Accuracy: ', initial_accuracy_train)

    initial_accuracy_dict = {'train': initial_accuracy_train, 'validation': initial_accuracy_validation}
    writer.add_scalars('Accuracy', initial_accuracy_dict)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('Loss/train', loss.item(), epoch*len(train_dataloader) + i)
            if i % 100 == 0:
                print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
                layer1_weights = model.conv1.weight.data
                grid_image = make_grid(layer1_weights, nrow=4, normalize=True, scale_each=True)
                writer.add_image(f'conv1_feature_maps_epoch_{epoch}_iteration_{i}', grid_image)
        
        # evaluate the model on the training and validation set and log the results
        train_accuracy, _, _, _, train_loss = evaluate(model, train_dataloader)
        validation_accuracy, _, _, _, validation_loss = evaluate(model, validation_dataloader)
        print('Train Accuracy: ', train_accuracy.item())
        loss_dict = {'train': train_loss.item(), 'validation': validation_loss.item()}
        accuracy_dict = {'train': train_accuracy.item(), 'validation': validation_accuracy.item()}

        writer.add_scalars('Loss', loss_dict, epoch*len(train_dataloader))
        writer.add_scalars('Accuracy', accuracy_dict, epoch*len(train_dataloader))
        
    print('Finished Training')

def evaluate(model, data_loader, loss_fn=nn.CrossEntropyLoss()):
    loss = torch.zeros(1)

    confusion_matrix = torch.zeros(10, 10)
    # determine the confusion matrix for the data_loader
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model.forward(images)
            l = loss_fn(outputs, labels).reshape(1)
            loss = torch.cat((loss, l))
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    

    # determine the overall accuracy using the confusion matrix
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()

    # determine the precision and recall for each class
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)

    return accuracy, confusion_matrix, precision, recall, torch.mean(loss)

def plot_images(model, training_data):
    lossfn = nn.CrossEntropyLoss(reduction='none')
    train_dataloader = DataLoader(training_data, batch_size=45000)
    images, labels = next(iter(train_dataloader))
    outputs = model.forward(images)
    loss = lossfn(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    # plot the 20 incorrectly classified images with the largest loss
    incorrect_images = images[predicted != labels]
    incorrect_labels = labels[predicted != labels]
    incorrect_predicted = predicted[predicted != labels]
    incorrect_loss = loss[predicted != labels]
    _, indices = torch.sort(incorrect_loss, descending=True)
    incorrect_images = incorrect_images[indices[:20]]
    incorrect_labels = incorrect_labels[indices[:20]]
    incorrect_predicted = incorrect_predicted[indices[:20]]
    # plot the images and label with the true and predicted label
    incorrect_images = incorrect_images.permute(0, 2, 3, 1)
    incorrect_images = incorrect_images.numpy()
    incorrect_labels = incorrect_labels.numpy()
    incorrect_predicted = incorrect_predicted.numpy()
    
    fig = plt.figure()
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1)
        ax.imshow(incorrect_images[i])
        ax.set_title(f'True: {incorrect_labels[i]}, Predicted: {incorrect_predicted[i]}')
        ax.axis('off')
    plt.show()


def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):  
    """
        Args:
            logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
            target: torch.LongTensor with shape (B, ) representing ground truth labels.
            delta: Hyperparameter.
        Returns:
            Loss as scalar torch.Tensor.
    """
    batch_size = logits.shape[0]
    correct_logits = logits[torch.arange(batch_size), target]
    correct_logits = correct_logits.reshape(-1, 1)
    correct_logits = correct_logits.repeat(1, logits.shape[1])
    temp = logits - correct_logits + delta
    temp[torch.arange(batch_size), target] = 0
    loss = torch.sum(torch.max(torch.zeros_like(logits), temp), dim=1)
    loss = torch.mean(loss)
    return loss


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


if __name__ == "__main__":
    # #=====================================================

    # DATA_DIR = 'CNNs/cifar10/cifar-10-batches-py'

    # img_height = 32
    # img_width = 32
    # num_channels = 3
    # num_classes = 10

    # train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    # train_y = []
    # for i in range(1, 6):
    #     subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    #     train_x = np.vstack((train_x, subset['data']))
    #     train_y += subset['labels']
    # train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    # train_y = np.array(train_y, dtype=np.int32)

    # subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    # test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    # test_y = np.array(subset['labels'], dtype=np.int32)

    # valid_size = 5000
    # train_x, train_y = shuffle_data(train_x, train_y)
    # valid_x = train_x[:valid_size, ...]
    # valid_y = train_y[:valid_size, ...]
    # train_x = train_x[valid_size:, ...]
    # train_y = train_y[valid_size:, ...]
    # data_mean = train_x.mean((0, 1, 2))
    # data_std = train_x.std((0, 1, 2))

    # train_x = (train_x - data_mean) / data_std
    # valid_x = (valid_x - data_mean) / data_std
    # test_x = (test_x - data_mean) / data_std

    # train_x = train_x.transpose(0, 3, 1, 2)
    # valid_x = valid_x.transpose(0, 3, 1, 2)
    # test_x = test_x.transpose(0, 3, 1, 2)

    # train_x = torch.from_numpy(train_x)
    # valid_x = torch.from_numpy(valid_x)
    # test_x = torch.from_numpy(test_x)
    # train_y = (torch.from_numpy(train_y)).long()
    # valid_y = (torch.from_numpy(valid_y)).long()
    # test_y = (torch.from_numpy(test_y)).long()

    # training_data = TensorDataset(train_x, train_y)
    # validation_data = TensorDataset(valid_x, valid_y)
    # test_data = TensorDataset(test_x, test_y)
    # =====================================================
    # training_data = datasets.CIFAR10(root="CNNs/cifar10",
    #                                   train=True, download=True, transform=ToTensor())
    
    
    # mean_values = torch.mean(training_data, axis=(0, 1, 2))
    # std_values = torch.std(training_data, axis=(0, 1, 2))

    

    # print("Mean values for each channel:", mean_values)
    # print("Std values for each channel:", std_values)


    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    training_data = datasets.CIFAR10(root="CNNs/cifar10",
                                      train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="CNNs/cifar10",
                                    train=False, download=True, transform=transform)
    
    
    # separate 5000 examples from the training set to create a validation set randomly
    training_data, validation_data = torch.utils.data.random_split(training_data, [0.9, 0.1])
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data)

    # for images, labels in train_dataloader:
    #     mean = torch.mean(images, axis=(0, 2, 3))
    #     std = torch.std(images, axis=(0, 2, 3))
    
    # print("Mean values for each channel:", mean)
    # print("Std values for each channel:", std)

    # pdb.set_trace()


    lr = 0.001 #0.001
    weight_decay = 0.001
    epochs = 10


    model = PTCIFAR()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = multiclass_hinge_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(comment=f'PTCIFAR_lr={lr},weight_decay={weight_decay}')
    train(model, train_dataloader, validation_dataloader, epochs, optimizer, loss_fn, writer)
    writer.close()
    accuracy, confusion_matrix, precision, recall, loss = evaluate(model, test_dataloader)
    print(f'Test Accuracy: {accuracy}')
    print(f'Confusion Matrix: {confusion_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    # plot_images(model, training_data)
    
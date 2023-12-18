import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

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
    
def train(model, train_dataloader, epochs, optimizer, loss_fn, writer):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch*len(train_dataloader) + i)
            if i % 100 == 0:
                print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
                layer1_weights = model.conv1.weight.data
                grid_image = make_grid(layer1_weights, nrow=4, normalize=True, scale_each=True)
                writer.add_image(f'conv1_feature_maps_epoch_{epoch}_iteration_{i}', grid_image)
        
    print('Finished Training')

def evaluate(model, data_loader):
    confusion_matrix = torch.zeros(10, 10)
    # determine the confusion matrix for the data_loader
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    

    # determine the overall accuracy using the confusion matrix
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()

    # determine the precision and recall for each class
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)

    return accuracy, confusion_matrix, precision, recall
    


if __name__ == "__main__":

    training_data = datasets.CIFAR10(root="CNNs/cifar10",
                                      train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR10(root="CNNs/cifar10",
                                    train=False, download=True, transform=ToTensor())
    # separate 5000 examples from the training set to create a validation set randomly
    training_data, validation_data = torch.utils.data.random_split(training_data, [0.9, 0.1])
    
    # normalize the data
    # train_mean = training_data.data.mean()
    # train_std = training_data.data.std()
    # training_data.data = (training_data.data - train_mean) / train_std
    # test_data.data = (test_data.data - train_mean) / train_std

    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_data)
    test_dataloader = DataLoader(test_data)

    model = PTCIFAR()
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter('CNNs/logs/cifar10')
    train(model, train_dataloader, epochs, optimizer, loss_fn, writer)
    accuracy, confusion_matrix, precision, recall = evaluate(model, test_dataloader)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix: {confusion_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    writer.close()

    # for images, labels in train_dataloader:
    #     # show image using matplotlib
    #     grid = make_grid(images, nrow=8)
    #     plt.figure()
    #     plt.imshow(np.transpose(grid, (1,2,0)))
    #     plt.axis('off')
    #     plt.show()
    #     break
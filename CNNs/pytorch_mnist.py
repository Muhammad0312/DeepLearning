import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import pdb
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


class PTMNIST(nn.Module):
    def __init__(self):
        super(PTMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding='same', bias=True)
        # output feature map size is 28x28x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #  out is 14x14x16
        self.relu1 = nn.ReLU()
        # out is 14x14x16
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding='same', bias=True)
        # out is 14x14x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # out is 7x7x32
        self.relu2 = nn.ReLU()
        # out is 7x7x32
        self.flatten3 = nn.Flatten()
        # out is 32*7*7
        self.fc3 = nn.Linear(32*7*7, 512, bias=True)
        # out is 512
        self.relu3 = nn.ReLU()
        # out is 512
        self.logits = nn.Linear(512, 10, bias=True)
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
        x = self.logits(x)
        return x

def train(model, train_dataloader, epochs, optimizer, loss_fn, writer):
    print('Starting Training')
    print('Initial Accuracy: ', evaluate(model, train_dataloader)[0].item())
    writer.add_scalar('Accuracy/train', evaluate(model, train_dataloader)[0])
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
        
        accuracy = evaluate(model, train_dataloader)[0].item()
        print('Train Accuracy: ', accuracy)
        writer.add_scalar('Accuracy/train', accuracy, epoch*len(train_dataloader))
        
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
    training_data = datasets.MNIST(root='FCNNs/mnist', 
        train=True, download=True, transform=ToTensor())

    test_data = datasets.MNIST(root='FCNNs/mnist',
        train=False, download=True, transform=ToTensor())
    

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data)

    model = PTMNIST()

    # add regularisation to cross entropy loss
    weight_decay = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    epochs = 10

    writer = SummaryWriter(comment=f'PTMNIST_lr=0.001,weight_decay={weight_decay}')
    train(model, train_dataloader, epochs, optimizer, loss_fn, writer)
    writer.close()


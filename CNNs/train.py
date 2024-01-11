import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

import nn
import layers

# DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
# SAVE_DIR = Path(__file__).parent / 'out'

DATA_DIR = 'FCNNs/mnist'
SAVE_DIR = 'CNNs/mnist/out'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))


net = []
inputs = np.random.randn(config['batch_size'], 1, 28, 28) 
# Tensor size (batch_size, channels, height, width) (50, 1, 28,28)
net += [layers.Convolution(inputs, 16, 5, "conv1")] 
# Tensor size (batch_size, channels, height, width) (50, 16, 28, 28)
# Params = (5*5*1+1)*16 = 416
net += [layers.MaxPooling(net[-1], "pool1")] 
# Tensor size (batch_size, channels, height, width) (50, 16, 14, 14)
net += [layers.ReLU(net[-1], "relu1")] 
# Tensor size (batch_size, channels, height, width) (50, 16, 14, 14)
net += [layers.Convolution(net[-1], 32, 5, "conv2")] 
# Tensor size (batch_size, channels, height, width) (50, 32, 14, 14)
# Params = (5*5*16+1)*32 = 12832
net += [layers.MaxPooling(net[-1], "pool2")] 
# Tensor size (batch_size, channels, height, width) (50, 32, 7, 7)
net += [layers.ReLU(net[-1], "relu2")] 
# Tensor size (batch_size, channels, height, width) (50, 32, 7, 7)
net += [layers.Flatten(net[-1], "flatten3")] 
# Tensor size (batch_size, outputs) (50, 1568)
net += [layers.FC(net[-1], 512, "fc3")] 
# Tensor size (batch_size, outputs) (50, 512)
# Params = (1568+1)*512 = 803328
net += [layers.ReLU(net[-1], "relu3")] 
# Tensor size (batch_size, outputs) (50, 512)
net += [layers.FC(net[-1], 10, "logits")] 
# Tensor size (batch_size, outputs) (50, 10)
# Params = (512+1)*10 = 5130

loss = layers.SoftmaxCrossEntropyWithLogits()

nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
nn.evaluate("Test", test_x, test_y, net, loss, config)

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
from model import SimpleMetricEmbedding, _BNReluConv

dataset = torchvision.datasets.MNIST("FCNNs/mnist", train=True, download=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
images = dataset.data.float() / 255.
targets = dataset.targets



x = images[:10].unsqueeze(1)

model = SimpleMetricEmbedding(1, 32)

y = model.get_features(x)
pdb.set_trace()
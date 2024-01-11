import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import NLPDataset, pad_collate_fn
import pdb

class BaseLine(nn.Module):
    def __init__(self, batch_size):
        super(BaseLine, self).__init__()
        self.batch_size = batch_size
        self.avg_pool = nn.AdaptiveAvgPool1d(batch_size)
        self.fc1 = nn.Linear(300, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)

    def forward(self,x):
        pdb.set_trace()
        x = self.avg_pool(x)
        x = x.squeeze()
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(model, dataloader, epochs, optimizer, loss_fn):
    print('Starting Training')
    model.train()
    for iteration in range(epochs):
        for i, (words, labels, lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model.forward(words)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Iteration: ', iteration, 'Loss: ', loss.item())
    print('Finished Training')

if __name__ == '__main__':
    batch_size = 10 # Only for demonstrative purposes
    shuffle = True # Only for demonstrative purposes
    train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    

    # for i, (words, labels, lengths) in enumerate(train_dataloader):
    #     print(words)
    #     print(labels)
    #     print(lengths)
    #     break
    model = BaseLine(batch_size=batch_size)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_dataloader, 1000, optimizer, loss_fn)
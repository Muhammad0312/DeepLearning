import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import *
import pdb
import numpy as np

class BaseLine(nn.Module):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.fc1 = nn.Linear(300, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(150, 1)

    def forward(self,x):
        x = x.sum(axis=1)/(x != 0).sum(axis=1).clamp(min=1).float()
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(model, dataloader, epochs, optimizer, loss_fn, embeddings, validation_dataloader):
    print('Starting Training')
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for i, (words, labels, lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            words = words.type(torch.LongTensor)
            words = embeddings(words)
            output = model.forward(words).squeeze()
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if i % 100 == 0:
                # print('Epoch: ', epoch, 'Batch: ', i, 'Loss: ', loss.item())
        # print('Epoch: ', epoch, 'Loss: ', total_loss/len(dataloader))
        accuracy, CM, _, _ = evaluate(model, validation_dataloader, loss_fn, embeddings)
        print('Epoch ', epoch, ': Valid accuracy: ', accuracy.item())
        # print(CM)
    print('Finished Training')

def evaluate(model, dataloader, loss_fn, embeddings):
    model.eval()
    confusion_matrix = torch.zeros(2, 2)
    # determine the confusion matrix for the data_loader
    with torch.no_grad():
        for i, (words, labels, lengths) in enumerate(dataloader):
            words = words.type(torch.LongTensor)
            words = embeddings(words)
            output = model.forward(words).squeeze()
            # loss = loss_fn(output, labels)
            predictions = torch.round(torch.sigmoid(output))
            for i in range(len(predictions)):
                confusion_matrix[int(labels[i]), int(predictions[i])] += 1
    
    # determine the overall accuracy using the confusion matrix
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()

    # determine the precision and recall for each class
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)

    return accuracy, confusion_matrix, precision, recall

if __name__ == '__main__':
    seed=7052020
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = 10
    shuffle = True
    train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv', train=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    
    text_vocab = Vocab(train_dataset.text_frequencies, max_size=-1, min_freq=0)
    label_vocab = Vocab(train_dataset.label_frequencies, max_size=-1, min_freq=0)
    
    test_dataset = NLPDataset('RNNs/data/sst_test_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    
    validation_dataset = NLPDataset('RNNs/data/sst_valid_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset), 
                                       collate_fn=pad_collate_fn)

    word_rep = load_embeddings('RNNs/data/sst_glove_6b_300d.txt')
    word_embeddings = gen_embeddings(text_vocab.stoi, word_rep)

    # pdb.set_trace()

    model = BaseLine()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train(model, train_dataloader, 5, optimizer, loss_fn, word_embeddings, validation_dataloader)
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import pdb


@dataclass
class Instance:
    words: list
    label: list

    def __iter__(self):
        yield self.words
        yield self.label

class NLPDataset(Dataset):
    instances: []
    word_frequencies = {}
    label_frequencies = {'positive': 0, 'negative': 1}
    def __init__(self, path):
        self.from_file(path)
        self.build_vocab()
        self.text_vocab = Vocab(self.word_frequencies, max_size=-1, min_freq=0)
        self.text_vocab.load_word_rep('RNNs/data/sst_glove_6b_300d.txt')
        self.label_vocab = Vocab(self.label_frequencies, max_size=-1, min_freq=0)


    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        words, label = self.instances[index]
        numericalized_words = self.text_vocab.encode(words)
        numericalized_label = self.label_vocab.encode(label)
        return numericalized_words, numericalized_label

    def from_file(self, file_path):
        instances = []
        with open(file_path, 'r') as f:
            for line in f:
                sentence, label = line.strip().split(',')
                words = sentence.split()
                instances.append(Instance(words, [label.strip()]))
        self.instances = instances

    def build_vocab(self):
        for instance in self.instances:
            for word in instance.words:
                self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1
        # Sort the words by frequency
        self.word_frequencies = dict(sorted(self.word_frequencies.items(), key=lambda item: item[1], reverse=True))
        char_dict = {'<PAD>': 0, '<UNK>': 1}
        self.word_frequencies = {**char_dict, **self.word_frequencies}
    
def pad_collate_fn(batch, pad_idx=0):
    # batch is a list of tuples (words, label)
    # words is a tensor of size (seq_len)
    # label is a tensor of size (1)
    # return padded_words, labels and lengths of original sequences
    batch_size = len(batch)
    max_seq_len = max([len(instance[0]) for instance in batch])
    padded_words = torch.zeros(batch_size, max_seq_len)
    labels = torch.zeros(batch_size)
    lengths = torch.zeros(batch_size)
    for i, instance in enumerate(batch):
        words, label = instance
        padded_words[i, :len(words)] = words
        labels[i] = label
        lengths[i] = len(words)
    return padded_words, labels, lengths


class Vocab:
    def __init__(self, frequencies, max_size = -1, min_freq = 0):
        self.freqs = frequencies
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = {}
        self.word_rep = {}
        self.embeddings = []
        self.make_stoi()
        self.make_itos()

    def make_stoi(self):
        for item in self.freqs.items():
            if self.max_size != -1 and len(self.stoi.item()) >= self.max_size:
                break
            word, freq = item
            if freq >= self.min_freq:
                self.stoi[word] = len(self.stoi)
    
    def make_itos(self):
        for word, index in self.stoi.items():
            self.itos[index] = word
            
    def encode(self, words):
        return torch.tensor([self.stoi.get(word, 0) for word in words])
    
    
    def load_word_rep(self, path):
        with open(path, 'r') as f:
            for line in f:
                word, vector = line.strip().split(' ', 1)
                vector = torch.tensor([float(num) for num in vector.split()])
                self.word_rep[word] = vector
        # self.word_rep['<UNK>'] = torch.zeros(300)
        self.gen_embeddings()
        
    
    def gen_embeddings(self):
        # generate a vector with random normal distribution of size len(stoi) x 300
        self.embeddings = torch.randn(len(self.stoi), 300)
        self.embeddings[0] = torch.zeros(300)
        for word, index in self.stoi.items():
            if word in self.word_rep and index != 0:
                self.embeddings[index] = self.word_rep[word]
        
        self.embeddings = nn.Embedding.from_pretrained(self.embeddings, freeze=True, padding_idx=0)



if __name__ == '__main__':
    batch_size = 1 # Only for demonstrative purposes
    shuffle = False # Only for demonstrative purposes
    train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    pdb.set_trace()
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")        






















    # text_vocab = Vocab(train_dataset.word_frequencies, max_size=-1, min_freq=0)
    # print(text_vocab.stoi)
    # label_vocab = Vocab(train_dataset.label_frequencies, max_size=-1, min_freq=0)

    # instance_text, instance_label = train_dataset.instances[3]
    # print(instance_text)
    # print(instance_label)
    # print(f"Numericalized text: {text_vocab.encode(instance_text)}")
    # print(f"Numericalized label: {label_vocab.encode(instance_label)}")
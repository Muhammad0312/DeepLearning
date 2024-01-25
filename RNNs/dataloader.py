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
    def __init__(self, path, train, text_vocab=None, label_vocab=None):
        self.train = train
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
        self.from_file(path)
        # if self.train:
        self.make_frequencies()

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        # return numericalized_words, numericalized_label where each word corresponds to the index in the vocabulary
        # and the vocabulary is sorted by frequency
        if self.train:
            words, label = self.instances[index]
            numericalized_words = self.words2idx(words)
            numericalized_label = self.label2idx(label)
            return numericalized_words, numericalized_label
        else:
            words, label = self.instances[index]
            numericalized_words = self.text_vocab.encode(words)
            numericalized_label = self.label_vocab.encode([label])
            return numericalized_words, numericalized_label
    
    def words2idx(self, words):
        sorted_words = sorted(self.text_frequencies, key=self.text_frequencies.get, reverse=True)
        idx = [sorted_words.index(word) for word in words]
        # add 2 to each index to account for padding and unknown tokens
        idx = [i + 2 for i in idx]
        return torch.tensor(idx)

    def label2idx(self, label):
        if label == 'positive':
            return torch.tensor(0)
        elif label == 'negative':
            return torch.tensor(1)
        
    def from_file(self, file_path):
        instances = []
        with open(file_path, 'r') as f:
            for line in f:
                sentence, label = line.strip().split(',')
                words = sentence.split()
                instances.append(Instance(words, label.strip()))
        self.instances = instances

    def make_frequencies(self):
        frequency = {}
        for instance in self.instances:
            for word in instance.words:
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
        
        #sort the dictionary based on the frequency of the words
        frequency = dict(sorted(frequency.items(), key=lambda item: item[1], reverse=True))
        self.text_frequencies = frequency
        self.label_frequencies = {'positive': 0, 'negative': 1}

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
        self.max_size = max_size
        self.min_freq = min_freq
        self.freqs = frequencies
        self.stoi = {}
        self.itos = {}
        self.word_rep = {}
        self.embeddings = []
        self.make_stoi()
        self.make_itos()

    def make_stoi(self):
        # only add pad and unk token size of self.freq is greater than 2
        if len(self.freqs) > 2:
            self.stoi = {'<PAD>': 0, '<UNK>': 1}
        
        for item in self.freqs.items():
            # if self.max_size != -1 and len(self.stoi.item()) >= self.max_size:
            #     break
            word, freq = item
            self.stoi[word] = len(self.stoi)
    
    def make_itos(self):
        for word, index in self.stoi.items():
            self.itos[index] = word
    
    def encode(self, words):
        return torch.tensor([self.stoi.get(word, 1) for word in words])

def load_embeddings(path):
    word_rep = {}
    with open(path, 'r') as f:
        for line in f:
            word, vector = line.strip().split(' ', 1)
            vector = torch.tensor([float(num) for num in vector.split()])
            word_rep[word] = vector
    #word_rep['<UNK>'] = torch.zeros(300)
    return word_rep

def gen_embeddings(vocabulary, word_rep):
    # generate a vector with random normal distribution of size len(stoi) x 300
    embeddings = torch.randn(len(vocabulary), 300)
    embeddings[0] = torch.zeros(300)
    for word, index in vocabulary.items():
        # print('word: ', word, 'index: ', index)
        if word in word_rep and index != 0:
            # print(f'Word {word} is in word_rep at index {index}')
            embeddings[index] = word_rep[word]
        
    embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
    return embeddings


if __name__ == '__main__':
    train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv')
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    instance_text, instance_label = train_dataset.instances[3]
    # We reference a class attribute without calling the overriden method
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")

    numericalized_text, numericalized_label = train_dataset[3]
    # We use the overriden indexing method
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")

    # batch_size = 2
    # shuffle = False
    # train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv')
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
    #                             shuffle=shuffle, collate_fn=pad_collate_fn)
    # texts, labels, lengths = next(iter(train_dataloader))

    # text_vocab = Vocab(train_dataset.text_frequencies, max_size=-1, min_freq=0)
    # label_vocab = Vocab(train_dataset.label_frequencies, max_size=-1, min_freq=0)

    # word_rep = load_embeddings('RNNs/data/sst_glove_6b_300d.txt')
    # word_embeddings = gen_embeddings(text_vocab.stoi, word_rep)
    # pdb.set_trace()

    # batch_size = 10
    # shuffle = False
    # train_dataset = NLPDataset('RNNs/data/sst_train_raw.csv', train=True)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
    #                             shuffle=shuffle, collate_fn=pad_collate_fn)
    
    # text_vocab = Vocab(train_dataset.text_frequencies, max_size=-1, min_freq=0)
    # label_vocab = Vocab(train_dataset.label_frequencies, max_size=-1, min_freq=0)
    
    # test_dataset = NLPDataset('RNNs/data/sst_test_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
    #                             shuffle=shuffle, collate_fn=pad_collate_fn)
    
    # validation_dataset = NLPDataset('RNNs/data/sst_valid_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    # validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset), 
    #                                    collate_fn=pad_collate_fn)
    
    
    

    # word_rep = load_embeddings('RNNs/data/sst_glove_6b_300d.txt')
    # word_embeddings = gen_embeddings(text_vocab.stoi, word_rep)

    pdb.set_trace()

    # train_keys = list(train_dataset.text_frequencies.keys())
    # train_numericalized_text = text_vocab.encode(train_keys)
    # train_vectorized_text = word_embeddings(train_numericalized_text)
    # data = list(zip(train_keys, train_numericalized_text.tolist(), train_vectorized_text.tolist()))
    # with open('RNNs/data/train_numericalized_text.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)
    # zip the test keys and the numericalized text and write to a text file
    # with open('RNNs/data/train_numericalized_text.txt', 'w') as f:
    #     for key, value, vector in zip(train_keys, train_numericalized_text, train_vectorized_text):
    #         f.write(f'{key} {value} {vector.tolist()}\n')
    
    # validation_keys = list(validation_dataset.text_frequencies.keys())
    # valid_numericalized_text = text_vocab.encode(validation_keys)
    # valid_vectorized_text = word_embeddings(valid_numericalized_text)
    # data = list(zip(validation_keys, valid_numericalized_text.tolist(), valid_vectorized_text.tolist()))
    # with open('RNNs/data/valid_numericalized_text.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)
    # zip the validation keys and the numericalized text and write to a text file
    # with open('RNNs/data/valid_numericalized_text.txt', 'w') as f:
    #     for key, value, vector in zip(validation_keys, valid_numericalized_text, valid_vectorized_text):
    #         f.write(f'{key} {value} {vector.tolist()}\n')


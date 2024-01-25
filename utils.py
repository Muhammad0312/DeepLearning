#!/usr/bin/python3

import torch
from torch import nn
from pathlib import Path
import os
# from torch.utils.tensorboard import SummaryWriter
import skimage as ski
import math
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import pdb
import skimage as ski
import skimage.io


from dataclasses import dataclass
from typing import List

import csv


from collections import Counter



@dataclass
class Instance:
    text: List[str]
    label: str


class NLPDataset(Dataset): # instances is a vocab object
    def __init__(self, text_vocab):
        self.text_vocab = text_vocab

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        instance_text  =  self.data_label[idx].text
        instance_label =  self.data_label[idx].label
        encoded_text   =  self.text_vocab.encode(instance_text)         # Encodes the text using the provided text vocabulary.
        encoded_label  =  self.text_vocab.encode(instance_label)       
        return encoded_text, encoded_label

    def from_file(self, file_path):
        self.data_label, _, _ = load_data_from_csv(file_path)

        dataset = []
        for i in range(len(self.data_label)):
            text, label = self.__getitem__(i)
            dataset.append((text, label))
        return dataset

class Vocab:
    """The frequency dictionary contains all the tokens that have appeared in that 
    field, with values representing the order of each token."""
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.stoi = {"<PAD>": 0, "<UNK>": 1}  # Initialize with special tokens
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.label_stoi = {"positive": 0, "negative": 1}  # Dictionary for label vocabulary
        self.frequencies = frequencies
        self.build_vocab(frequencies, max_size, min_freq)

    def build_vocab(self, frequencies, max_size, min_freq): # build vocabulary for that instance or field
        # Sort by frequency and index tokens based on frequency
        sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        sorted_tokens = [token[0] for token in sorted_tokens if token[1] >= min_freq]
        if max_size > 0:    # Limits the vocabulary size if max_size is specified.
            sorted_tokens = sorted_tokens[:max_size - len(self.stoi)]
        for idx, token in enumerate(sorted_tokens, len(self.stoi)):
            self.stoi[token] = idx
            self.itos[idx] = token

    def encode(self, tokens):   # Method to convert tokens=sentence or word to their corresponding indices.
        # pdb.set_trace()
        
        if isinstance(tokens, list):  # Assuming list input for text tokens
            return torch.tensor([self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens])
            # Returns a list of indices for each token in tokens. If a token is not 
            # found in the vocabulary, it returns the index of the <UNK> token.
        
        else:  # Handling label tokens
            return torch.tensor(self.label_stoi.get(tokens))
        
# # # print(len(text_vocab.itos)) # total number of words
def generate_embedding_matrix(vocab, embedding_file_path, embed_dim=300): # vocab is a Vocab class instance (sentence)
    vocab_size = len(vocab.stoi)
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embed_dim)) # (vocab_size, embed_dim) = (v, d)
    with open(embedding_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0] # Retrieves the word/token.
            if word in vocab.stoi:
                idx = vocab.stoi[word]
                vector = np.array(values[1:], dtype='float32')
                embedding_matrix[idx] = vector        
    embedding_matrix[0] = np.zeros(embed_dim)  # Padding token
    embedding_matrix_tensor = torch.tensor(embedding_matrix) #(14806, 300)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix_tensor, padding_idx=0, freeze=True)
    # pdb.set_trace()
    return embedding_layer
    # return torch.tensor(embedding_matrix)


def pad_collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, torch.tensor(labels), lengths

# Function to load data from a CSV file
def load_data_from_csv(file_path):
    instances = []
    text_frequencies = Counter()
    label_frequencies = Counter()
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            text, label = row[0], row[1][1:]
            instances.append(Instance(text=text.split(), label=label))
            text_frequencies.update(text.split())
            label_frequencies.update([label])
    return instances, text_frequencies, label_frequencies


def save_frequencies(train_instances):
    """ Loading the frequencies """
    # Assuming train_instances is a list of Instance objects with 'text' attribute as a list of words
    train_text = [instance.text for instance in train_instances]
    # Flatten the list of lists into a single list of words
    flattened_text = [word for sublist in train_text for word in sublist]
    # Get word frequencies
    word_frequencies = Counter(flattened_text)

    # Now, word_frequencies contains the count of each word in the training data
    with open('data/frequencies.pkl', 'wb') as file:
        pickle.dump(word_frequencies, file)

def load_train_frequencies():
    """ Extracting the frequencies """
    frequencies_dict = {}
    with open('data/frequencies.pkl', 'rb') as file:
        frequencies_dict = pickle.load(file)
    return frequencies_dict

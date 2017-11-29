import os
import torch
import csv

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, type='aqua',bpe=False):
        self.dictionary = Dictionary()

        if type == 'aqua':
            if bpe==False:
                self.train = self.tokenize(os.path.join(path, 'train.csv'))
                self.valid = self.tokenize(os.path.join(path, 'dev.csv'))
                self.test = self.tokenize(os.path.join(path, 'test.csv'))
            else:
                self.train = self.tokenize(os.path.join(path, 'train_bpe.csv'))
                self.valid = self.tokenize(os.path.join(path, 'dev_bpe.csv'))
                self.test = self.tokenize(os.path.join(path, 'test_bpe.csv'))

        else:
            if bpe==False:
                self.train = self.tokenize(os.path.join(path, 'mawps_train.csv'))
                self.valid = self.tokenize(os.path.join(path, 'mawps_valid.csv'))
                self.test = self.tokenize(os.path.join(path, 'mawps_test.csv'))
            else:
                self.train = self.tokenize(os.path.join(path, 'mawps_train_bpe.csv'))
                self.valid = self.tokenize(os.path.join(path, 'mawps_valid_bpe.csv'))
                self.test = self.tokenize(os.path.join(path, 'mawps_test_bpe.csv'))


    def tokenize(self, path):
        """Tokenizes a text file."""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary

        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = " ")
            tokens = 0
            for row in reader:
                words = row[0].split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = " ")
            ids = torch.LongTensor(tokens)
            token = 0
            for row in reader:
                words = row[0].split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
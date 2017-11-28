import os
import torch
import csv 
import numpy as np
import pickle
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self):
        self.unk_tok = '<unk>'
        self.sent_tok = '</s>'
        self.word2idx = { self.unk_tok : 0, self.sent_tok : 1 }
        self.idx2word = [ self.unk_tok, self.sent_tok ]
        self.unk_idx = self.word2idx[self.unk_tok]
        self.sent_idx = self.word2idx[self.sent_tok]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, key):
        if type(key) == str:
            return self.word2idx.get(key, self.unk_idx)
        elif type(key) == int:
            return self.idx2word[key]
        else:
            raise KeyError


class Corpus(object):
    def __init__(self, path, have_dict):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path,'Train','train.csv'))
        self.valid = self.tokenize(os.path.join(path,'Dev','dev.csv'))
        self.test = self.tokenize(os.path.join(path,'Test','test.csv'))
        if have_dict:
            self.dictionary = pickle.load(open('dic.pkl','rb'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        bsz = 64
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = " ")
            num_tokens = 0
            tokens = []
            for row in reader:
                words = row[0].split()
                num_tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                tokens += map(self.dictionary.__getitem__, words)
                strip_len = len(tokens) // bsz
                usable = strip_len * bsz
                data = np.asarray(tokens[:usable]).reshape(bsz, strip_len).transpose()
                seq_len = 35 #what is this?
                for b in range(strip_len // seq_len):
                    source = torch.LongTensor(data[(b*seq_len):((b+1)*seq_len), :]).contiguous() 
                    target = torch.LongTensor(data[(b*seq_len)+1:((b+1)*seq_len)+1, :]).contiguous() #+1 on first entry
                    if target.shape[0]< 35:
                         target = torch.LongTensor(data[(b*seq_len):((b+1)*seq_len)+1, :]).contiguous() 

                    #print(target.shape)
                    print(source.shape)
                    source = Variable(source, volatile=False)
                    target = Variable(target)
                    yield (source, target)

    def build_dictionary(self,path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = " ")
            for row in reader:
                words = row[0].split()
                for word in words:
                    self.dictionary.add_word(word)
        pickle.dump(self.dictionary, open('dic.pkl','wb'))

#corpus = Corpus('./Data',False) 
#corpus.build_dictionary('./Data/Train/train.csv')
#print(len(corpus.dictionary.idx2word))

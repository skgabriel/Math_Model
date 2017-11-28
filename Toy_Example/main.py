import sys, time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.insert(0, '../utils/')
from data import Dictionary 

import data
import model
import nltk 

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
corpus = data.Corpus('./Data',True)
ntokens = len(corpus.dictionary.idx2word)
model = model.RNNModel() #model, ntokens, embedding size, number of hidden layers, number of layers, dropout, tied
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

# Loop over epochs.
global lr, best_val_loss
lr = 20
best_val_loss = None

#Why?

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, nbatches = 0, 0
    ntokens = len(corpus.dictionary.idx2word) #2740
    bsz = 64
    hidden = model.init_hidden(bsz)
    for source, target in corpus.tokenize('./Data/Dev/dev.csv'):
        #print("Matrices")
        #print(hidden.shape)
        #print(source.shape)
        print(source.size())
        print(hidden.size())
        output, hidden = model(source, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += criterion(output_flat, target.view(-1)).data
        hidden = repackage_hidden(hidden)
        nbatches += 1
    return total_loss[0] / nbatches


def train():
    global lr, best_val_loss
    # Turn on training mode which enables dropout.
    print("About to begin training.")
    model.train()
    total_loss, nbatches = 0, 0
    start_time = time.time()
    ntokens = len(corpus.dictionary.idx2word)
    bsz = 64
    hidden = model.init_hidden(bsz)
    i = 0
    for b, batch in enumerate(corpus.tokenize('./Data/Train/train.csv')):
        i += 1
        print(i)
        source, target = batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        print(source.size())
        print(hidden.size())
        output, hidden = model(source, hidden)
        loss = criterion(output.view(-1, ntokens), target.view(-1))
        loss.backward()

        #why?
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(),.2)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        print(total_loss[0])

        print("evaluate")
        #corpus_valid = data.Corpus('dev',True)
        evaluate('dev')

#        if b % 1 == 0 and b > 0: #b % 500
#            cur_loss = total_loss[0] / 1 # / 500
#            elapsed = time.time() - start_time
#            val_loss = evaluate('valid')
 #           print("where does it get stuck?")
 #           print('| epoch {:3d} | batch {:5d} | lr {:02.5f} | ms/batch {:5.2f} | '
#                    'loss {:5.2f} | ppl {:8.2f} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
 #               epoch, b, lr,
#                elapsed * 1000 / 1, cur_loss, math.exp(cur_loss), # / 500
 #               val_loss, math.exp(val_loss)))
#
          #  # Save the model if the validation loss is the best we've seen so far.
          #  if not best_val_loss or val_loss < best_val_loss:
          #      with open('best_model.pt', 'wb') as f:
           #         torch.save(model, f)
           #     best_val_loss = val_loss
           # else:
                ## Anneal the learning rate if no improvement has been seen in the validation dataset. #why?
                #lr *= .1
                #total_loss = 0
                #start_time = time.time()



# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, 10+1): #10 epochs 
        epoch_start_time = time.time()
        train()
        val_loss = evaluate('valid')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('best_model.pt', 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate('test')
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
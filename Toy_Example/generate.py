import torch
from torch.autograd import Variable

import data

d='./Data/Test'
checkpoint = './best_model.pt'
outf = 'generated.txt'
words = 1000
temperature = 1.0 
log-interval = 100 

# Set the random seed manually for reproducibility.

torch.manual_seed(1111)
#torch.cuda.manual_seed(1111)

with open(checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
#model.cuda()
model.cpu()

corpus = data.Corpus(d)
corpus.build_dictionary() 
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
#input.data = input.data.cuda()

with open(outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
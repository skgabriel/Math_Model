# coding: utf-8

# In[20]:

import json
import csv 
import argparse
from nltk.tokenize import sent_tokenize
from io import open
import sys

def convert_text(fobj,output_name):

    data = [] 
    for line in fobj:
        data.append(json.loads(line))
    questions = [] 
    for item in data:
        questions.append(item['question'])
    tokenized_questions = [] 
    for q in questions:
        tokenized_questions.append(sent_tokenize(q))
    output = open(output_name, 'w')
    writer = csv.writer(output, delimiter=" ")
    for document in tokenized_questions:
        row = " <bop> "
        for question in document:
            row += str(question.encode('utf8').lower())
            row += " <eos> "
        writer.writerow([row])
    output.close() 

def main():
    convert_text(open('./Data/Train/train.json','r'),'./Data/Train/train.csv')

if __name__ == '__main__':

   # parser = create_parser()
   # args = parser.parse_args()

    # read/write files as UTF-8
    #if args.input.name != '<stdin>':
    #    args.input = codecs.open(args.input.name, encoding='utf-8')

    main()

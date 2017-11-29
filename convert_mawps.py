import json
import csv 
import argparse
from nltk.tokenize import sent_tokenize
from io import open
import sys
import sklearn as skl

def convert_text(fobj):

    data = json.load(fobj)
    train, test = skl.model_selection.train_test_split(data,test_size=.1)
    train, valid = skl.model_selection.train_test_split(data,test_size=.1)
    gen_data(train,'./Data/mawps_train.csv')
    gen_data(valid, './Data/mawps_valid.csv')
    gen_data(test, './Data/mawps_test.csv')


def gen_data(data,output_name):
    questions = [] 

    for item in data:
        questions.append(item['sQuestion'])
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
    convert_text(open('./Data/AllData.json','r'))

if __name__ == '__main__':

   # parser = create_parser()
   # args = parser.parse_args()

    # read/write files as UTF-8
    #if args.input.name != '<stdin>':
    #    args.input = codecs.open(args.input.name, encoding='utf-8')

    main()

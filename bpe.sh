#!/bin/bash 

#./learn_bpe.py -s {num_operations} < {train_file} > {codes_file}
#./apply_bpe.py -c {codes_file} < {test_file}

./learn_bpe.py < ./Data/Train/train.csv > ./Data/Train/codes.txt 
./apply_bpe.py -c ./Data/Train/codes.txt < ./Data/Train/train.csv >  ./Data/Train/train_bpe.csv
#!/bin/bash 

#./learn_bpe.py -s {num_operations} < {train_file} > {codes_file}
#./apply_bpe.py -c {codes_file} < {test_file}

./learn_bpe.py < ./Data/train.csv > ./Data/codes.txt 
./learn_bpe.py < ./Data/mawps_train.csv > ./Data/codes_mawps.txt 

./apply_bpe.py -c ./Data/codes.txt < ./Data/train.csv >  ./Data/train_bpe.csv
./apply_bpe.py -c ./Data/codes_mawps.txt < ./Data/mawps_train.csv >  ./Data/mawps_train_bpe.csv

./apply_bpe.py -c ./Data/codes.txt < ./Data/dev.csv >  ./Data/dev_bpe.csv
./apply_bpe.py -c ./Data/codes_mawps.txt < ./Data/mawps_valid.csv >  ./Data/mawps_valid_bpe.csv

./apply_bpe.py -c ./Data/codes.txt < ./Data/test.csv >  ./Data/test_bpe.csv
./apply_bpe.py -c ./Data/codes_mawps.txt < ./Data/mawps_test.csv >  ./Data/mawps_test_bpe.csv
#!/bin/bash

python main.py --cuda --save 'model_aqua.pt' --log 'aqua_output.log'
python main.py --cuda --save 'model_mawps.pt' --mawps --log 'mawps_output.log'
python main.py --cuda --save 'model_aqua_bpe.pt' --bpe --log 'aqua_bpe_output.log'
python main.py --cuda --save 'model_mawps_bpe.pt'--mawps --bpe --log 'mawps_bpe_output.log'
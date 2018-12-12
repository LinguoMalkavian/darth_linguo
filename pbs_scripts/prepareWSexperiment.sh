

python corpus_tools.py  split_corpus new_hundo-base --named_corpus new_hundo --piece_names  --outfile new_hundo

python
python generateWS.py full_ws-LM1 --numsalads 605524 --named_corpus full_ws --suf train
python generateWS.py full_ws-LM2 --numsalads 151381 --named_corpus full_ws --suf val

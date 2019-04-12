#!/bin/bash 
N_HISTORY=$1 # First command line argument

echo "Generating training files...."
python3 scripts/preprocess_coqa.py \
    --src_path=./data \
    --type=train \
    --n_history=$N_HISTORY

echo "Generating dev files......"
python3 scripts/preprocess_coqa.py \
    --src_path=./data \
    --type=dev \
    --n_history=$N_HISTORY

echo "Performing dev/test split"
cd data/dev
mkdir -p ../test
mv coqg_h${N_HISTORY}_dev.para coqg_h${N_HISTORY}.para
mv coqg_h${N_HISTORY}_dev.ques coqg_h${N_HISTORY}.ques
head -n 4009 coqg_h${N_HISTORY}.para > ../test/coqg_h${N_HISTORY}_test.para
tail -n +4010 coqg_h${N_HISTORY}.para > coqg_h${N_HISTORY}_dev.para
head -n 4009 coqg_h${N_HISTORY}.ques > ../test/coqg_h${N_HISTORY}_test.ques
tail -n +4010 coqg_h${N_HISTORY}.ques > coqg_h${N_HISTORY}_dev.ques
rm coqg_h${N_HISTORY}.para
rm coqg_h${N_HISTORY}.ques
echo "Finished dev/test split"

cd ../..

echo "Preprocessing all done..."
echo "Now in directory `$pwd`"
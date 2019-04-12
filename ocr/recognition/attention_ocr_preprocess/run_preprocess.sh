#!/usr/bin/env bash
SIZE=$1

PHASE=$2

STOP_WORD_PER=$3

if [[ $PHASE = "train" ]]
then
    test=False
else
    test=True
fi

python preprocess_dataset.py --stop-words-per ${STOP_WORD_PER} --outputFile /data/aocr/tfrecords/${PHASE}_output.txt --test ${test} --starts 0 --sizes $SIZE --data-paths /data/crops

mkdir -p /data/aocr/tfrecords/${PHASE}/

python generate.py dataset /data/aocr/tfrecords/${PHASE}_output.txt /data/aocr/tfrecords/${PHASE}/
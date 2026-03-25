#! /usr/bin/env bash

logdir=./exp/make_mfcc
outdir=./mfcc


for x in train dev test; do
        ./steps/make_mfcc.sh --nj 1 ./data/$x $logdir $outdir
        ./steps/compute_cmvn_stats.sh  ./data/$x $logdir $outdir
done

echo "finished gracefully"

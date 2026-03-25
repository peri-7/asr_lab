#! /usr/bin/env bash

. ./path.sh || exit 1

echo "============ Question 1 =============="

lmdir=./data/local/lm_tmp
#lmdir=./data/local/nist_lm
dictdir=./data/local/dict

if [ ! -f $dictdir/lm_test.text ] ||  [ ! -f $dictdir/lm_dev.text ]; then
        echo "Files in dict not found";
        exit 1
fi

for lm in ug bg; do
        for data in test dev; do
                 echo "-> Model: ${lm} | Dataset: ${data}"
                 compile-lm ${lmdir}/lm_phone_${lm}.ilm.gz -e=${dictdir}/lm_${data}_tags.text 2>&1 | grep -i "PP"
                 #compile-lm ${lmdir}/lm_phone_${lm}.arpa.gz -e=${dictdir}/lm_${data}_tags.text 2>&1 | grep -i "PP"
                 #compile-lm ${lmdir}/lm_phone_${lm}.ilm.gz /dev/null -e=${dictdir}/lm_${data}.text 2>&1 | grep -i "PP"
        done
done

echo "Finished gracefully"

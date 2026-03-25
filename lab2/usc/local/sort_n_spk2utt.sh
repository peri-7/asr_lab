#! /usr/bin/env bash


source ./path.sh

for dir in train test dev; do
        for f in wav.scp text utt2spk; do
                sort ./data/$dir/$f -o ./data/$dir/$f
        done

        ./utils/utt2spk_to_spk2utt.pl ./data/$dir/utt2spk > ./data/$dir/spk2utt
done

echo "spk2utt created gracefully"

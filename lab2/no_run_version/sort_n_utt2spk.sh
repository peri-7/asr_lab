source ../path.sh

for dir in train test dev; do
        for f in wav.scp text utt2spk; do
                sort ./$dir/$f -o ./$dir/$f
        done

        ../utils/utt2spk_to_spk2utt.pl ./$dir/utt2spk > ./$dir/spk2utt
done

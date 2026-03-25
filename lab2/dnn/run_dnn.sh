#!/usr/bin/env bash

DATA_PATH=./data/test


# FIXME: CHANGE THESE PATHS TO MATCH YOUR CONFIG
GRAPH_PATH=./exp/tri1/graph_bg
TEST_ALI_PATH=./exp/tri1_ali_test
OUT_DECODE_PATH=./exp/tri1/decode_test_dnn


CHECKPOINT_FILE=./best_usc_dnn.pt
DNN_OUT_FOLDER=./dnn_out

# ------------------- Data preparation for DNN -------------------- #
# 4.5.1
# Triphone alignments for all datasets
steps/align_si.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali_train || exit 1
steps/align_si.sh --nj 1 --cmd "$train_cmd" data/dev data/lang exp/tri1 exp/tri1_ali_dev || exit 1
steps/align_si.sh --nj 1 --cmd "$train_cmd" data/test data/lang exp/tri1 exp/tri1_ali_test || exit 1

# 4.5.2
# Compute cmvn stats for every set and save them in specific .ark files
# These will be used by the python dataset class that you were given
for set in train dev test; do
  compute-cmvn-stats --spk2utt=ark:data/${set}/spk2utt scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_speaker.ark"
  compute-cmvn-stats scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_snt.ark"
done


# ------------------ TRAIN DNN ------------------------------------ #
# delete previous best model
if [ -f best_usc_dnn.pt ]; then
        rm best_usc_dnn.pt
fi
python timit_dnn.py $CHECKPOINT_FILE


# ----------------- EXTRACT DNN POSTERIORS ------------------------ #
python extract_posteriors.py $CHECKPOINT_FILE $DNN_OUT_FOLDER


# ----------------- RUN DNN DECODING ------------------------------ #
./decode_dnn.sh $GRAPH_PATH $DATA_PATH $TEST_ALI_PATH $OUT_DECODE_PATH "cat $DNN_OUT_FOLDER/posteriors.ark"
echo
echo "PER results:"
cat exp/tri1/decode_test_dnn/scoring_kaldi/best_wer
echo

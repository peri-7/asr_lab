#!/usr/bin/env bash

# Authors:
# -> Tsimplaki Pinelopi Anna (nelly)
# -> Alexiou Periklis (fiction)


# default stage value
stage=0
stop_stage=100

. ./cmd.sh
. ./path.sh

# for --stage i recognition
. utils/parse_options.sh


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo
  echo "==== STAGE 0: Data & Language Preparation ==="
  echo
  # uttids, utt2spk, wav.scp, text in data/train (etc) creation
  mkdir -p data/train data/dev data/test
  python ./local/prepare_data.py

  # prepare all the data/local/dict files
  mkdir -p data/lang data/local/dict data/local/lm_tmp data/local/nist_lm
  cp ./local/sources/lexicon.txt ./data/local/dict/word_lex.txt
  python ./local/prepare_dict.py

  # prepare data/lang and create utt2spk
  utils/prepare_lang.sh data/local/dict "sil" data/local/lang_tmp data/lang
  ./local/sort_n_spk2utt.sh
fi



if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo
  echo "==== STAGE 1: Language Moddel & G.fst Creation ===="
  echo
  # remove uttid for the training and the evaluation
  cut -d' ' -f2- data/local/dict/lm_train.text > data/local/dict/lm_train_tags.text
  cut -d' ' -f2- data/local/dict/lm_dev.text > data/local/dict/lm_dev_tags.text
  cut -d' ' -f2- data/local/dict/lm_test.text > data/local/dict/lm_test_tags.text

  # build the lm and compile it
  build-lm.sh -i data/local/dict/lm_train_tags.text -n 1 -o data/local/lm_tmp/lm_phone_ug.ilm.gz
  build-lm.sh -i data/local/dict/lm_train_tags.text -n 2 -o data/local/lm_tmp/lm_phone_bg.ilm.gz

  compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz
  compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz

  # build the G.fst
  ./local/usc_format_data.sh

  # print perplexity of lm
  ./local/questions/question1.sh
fi



if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo
  echo "==== STAGE 2: MFCC & CMVN (features) Extraction ===="
  echo
  # extract mfccs
  for x in train dev test; do
    steps/make_mfcc.sh --nj 1 data/$x exp/make_mfcc/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
  done

  echo "finished extracting MFCCs gracefully"

  # print frames per sentence and dimention of features
  ./local/questions/question3.sh
fi



if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo
  echo "==== STAGE 3: Mono Training ===="
  echo
  # GMM-HMM training
  steps/train_mono.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1

  echo "training finished gracefully"
fi



if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo
  echo "==== STAGE 4: Mono Decoding ===="
  echo

  for lm in ug bg; do
        utils/mkgraph.sh --mono data/lang_test_${lm} exp/mono exp/mono/graph_${lm} || exit 1

        for dataset in dev test; do
                steps/decode.sh --nj 1 --cmd "$decode_cmd" exp/mono/graph_${lm} data/${dataset} exp/mono/decode_${dataset}_${lm} || exit 1
        done
  done

  # to check if it s working, run "tail -f exp/mono/decode_test_ug/log/decode.1.log" (change dataset and model accordingly) in another shell

  echo
  echo "PER results:"
  for x in exp/mono/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  echo
  echo "decoding finished gracefully"
fi



if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo
  echo "==== STAGE 5: Mono Alignment / Tri1 Training & Decoding ---"
  echo

  echo
  echo "--- Mono Alignment ---"
  steps/align_si.sh --nj 1 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1

  echo
  echo "--- Tri1 Training ---"
  steps/train_deltas.sh --cmd "$train_cmd" 2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1

  echo
  echo "--- Tri1 Decoding ---"
  for lm in ug bg; do
    utils/mkgraph.sh data/lang_test_${lm} exp/tri1 exp/tri1/graph_${lm} || exit 1

    for dataset in dev test; do
      steps/decode.sh --nj 1 --cmd "$decode_cmd" \
        exp/tri1/graph_${lm} data/${dataset} exp/tri1/decode_${dataset}_${lm} || exit 1
    done
  done

  # to check if it s working, run "tail -f exp/tri1/decode_test_ug/log/decode.1.log" (change dataset and model accordingly) in another shell

  echo
  echo "PER results:"
  for x in exp/tri1/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  echo
  echo "triphone pipeline finished gracefully"



fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo
  echo "==== STAGE 6: DNN-HMM Training & Decoding ===="
  echo

  ./run_dnn.sh

  echo
  echo "dnn pipeline finished gracefully"
fi

#! /usr/bin/env bash


. ./path.sh

echo "============== Question 3 =============="

echo "frames per utterance:"
feat-to-len scp:data/train/feats.scp ark,t:- 2>/dev/null | head -n 5
echo "feature dimension:"
feat-to-dim scp:data/train/feats.scp - 2>/dev/null

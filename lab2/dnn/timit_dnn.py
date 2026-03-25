#
# 4.5.4Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from torch_dataset import TorchSpeechDataset
from torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]


# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = "exp/tri1_ali_train"
DEV_ALIGNMENT_DIR = "exp/tri1_ali_dev"
TEST_ALIGNMENT_DIR = "exp/tri1_ali_test"


def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    # TODO: IMPLEMENT THIS FUNCTION
    # 4.5.4

    best_vloss = float('inf')
    pat_cnt = 0

    for epoch in range(epochs):

        # training
        model.train()
        running_loss = 0

        train_data = tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}")
        for inputs, labels in train_data:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # evaluation
        model.eval()
        running_vloss = 0

        with torch.no_grad():
            val_data = tqdm(dev_loader, desc=f"Validation {epoch+1}/{epochs}")
            for vinputs, vlabels in val_data:
                vinputs = vinputs.to(DEVICE)
                vlabels = vlabels.to(DEVICE)

                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / len(dev_loader)
        print(f"Epoch: {epoch+1}\nLOSS: Train->{avg_loss} | Val->{avg_vloss}")

        # early stopping
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            pat_cnt = 0
            torch.save(model, BEST_CHECKPOINT)
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print("early stopping b!tch")
                break


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)


dnn = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
dnn.to(DEVICE)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

# 4.5.4
optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train(dnn, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)

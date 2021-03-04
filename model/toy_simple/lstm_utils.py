"""Utils related to lstm models and dataset vocab."""

import pandas as pd
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """
    Similar to transformer's, adapted to this dataset with Marc's line by line reading output
    """

    def __init__(self, data, max_len=35, pad_idx=0):
        self.data = data
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_lst = self.data[idx][2]
        if len(idx_lst) > self.max_len:
            idx_lst = idx_lst[: self.max_len]
        idx_lst = idx_lst + [self.pad_idx] * (self.max_len - len(idx_lst))

        return (
            self.data[idx][0],
            torch.tensor(self.data[idx][1]),
            torch.tensor(idx_lst),
        )


class Vocab:
    def __init__(self, vocab, rev_vocab):
        self._vocab = vocab
        self._rev_vocab = rev_vocab
        self._unk_idx = vocab["<unk>"]
        self._pad_idx = vocab["<pad>"]

    def __len__(self):
        return len(self._vocab)

    def stoi(self, token):
        if token in self._vocab:
            return self._vocab[token]
        return self._vocab["<unk>"]

    def itos(self, idx):
        if idx in self._rev_vocab:
            return self._rev_vocab[idx]
        return self._unk_idx


def build_lstm_dataset(
    datapath,
    min_freq=10,
    uid_colname="patient_id",
    target_colname="label",
    max_len=30,
    target_value="1",
    vocab=None,
    nrows=1e9,
    rev=True,
):
    """
    Reads in dataset line by line, creates vocabulary and subsets dataset to list of arrays to be used later.
    Note: Key difference to transformer build_dataset (has remove_death, build_vocab functions together):
        1. Line by line reading, as the 1000 format CSV file takes a long time to read via pandas
        2. Incorporates remove death immediately
        3. Builds vocabulary here

    Arguments:
    ---------
        datapath (str) : path to CSV file
        min_freq (int) : minimum frequency for vocabulary
        uid_colname (str) : name of unique identifier in each row (patient_id, discharge_dt)
                            returned in list of arrays
        target_colname (str) : column name of target label
        max_len(int): Number of input features
        target_value (str) : value to take as positive label
        vocab (Vocab) : default None creates vocabulary. Will use provided vocab if given.
        nrows (int) : number of rows to process
        rev(bool): Whether to reverse the token list

    Returns:
    --------
        Tuple containing:
            (dataset object) with a list of data and attributes in the following order:
                patient_id key,
                sequence of events,
                targets, and
            Vocab Object
    """

    def valid_token(t):
        if len(t) == 0 or t == "<pad>":
            return False
        return True

    print("Building dataset from {}..".format(datapath))
    x_inputs = [str(x) for x in range(max_len - 1, -1, -1)]

    token_list = []
    with open(datapath, "r") as f:
        # determine column mapping
        header = f.readline()
        header = [h.replace(" ", "").replace("\n", "") for h in header.split(",")]

        target_index = header.index(target_colname)

        uid_index = header.index(uid_colname)

        x_idxes = []
        for colname in x_inputs:
            x_idxes.append(header.index(colname))

        # start processing
        line = f.readline()
        invalid_uid = 0
        deaths = 0
        while line:
            if "death" in line:
                deaths += 1
                pass

            tokens = line.split(",")

            if len(tokens[uid_index]) == 0:
                invalid_uid += 1  # some UIDS are missing
                pass
            else:
                uid = tokens[uid_index].replace("\n", "")

            ## CHANGE: integer
            if isinstance(tokens[target_index], str):
                label = 1 if tokens[target_index].startswith(target_value) else 0

            if isinstance(tokens[target_index], int):
                label = tokens[target_index]

            tokens = [tokens[idx] for idx in x_idxes]
            tokens = [t.strip().replace("\n", "") for t in tokens if valid_token(t)]

            if rev:
                tokens = tokens[::-1]

            token_list.append((uid, label, tokens))

            line = f.readline()

            if len(token_list) == nrows:
                break

    if vocab is None:
        vocab = {}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1

        rev_vocab = {}
        rev_vocab[0] = "<pad>"
        rev_vocab[1] = "<unk>"

        counter = Counter()
        for uid, label, tokens in token_list:
            for token in tokens:
                counter[token] += 1

        for token in counter:
            if counter[token] < min_freq:
                continue

            idx = len(vocab)
            vocab[token] = idx
            rev_vocab[idx] = token
        vocab = Vocab(vocab, rev_vocab)

    data = [
        (pid, [label], [vocab.stoi(t) for t in tokens])
        for pid, label, tokens in token_list
    ]
    dataset = ToyDataset(data, max_len=max_len)
    print("Success!")
    return dataset, vocab


def epoch_time(start_time, end_time):
    """Calculate total amount fo time spent training"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def epoch_train_lstm(model, dataloader, optimizer, criterion, test=0, clip=False, device=None):
    """
    Train model for an epoch, called by ModelProcess function
    detach_hidden is used to detach hidden state between batches,
    and will add a new hidden state to model. Model must have .init_hidden function defined

    Args:
    -----
        model (nn.Module): lstm general attention model
        dataloader : iterator for dataset, yields (ids, sequence, seq length, labels)
        criterion : loss function
        optimizer : pytorch optimizer to be used during step
        clip (bool) : clip gradients if enabled
        
    Returns:
    ----------
        tuple containing:
            average loss for whole epoch,
            average AUC for whole epoch
    """
    epoch_loss = 0
    epoch_metric = 0

    model.train()
    optimizer.zero_grad()

    # initialize lists to compare predictions & ground truth labels for metric calculation
    order_labels = []
    prediction_scores = []
    if test:  # test function on small number of batches
        counter = 0
        
    for idx, (ids, labels, idxed_text) in enumerate(dataloader):

#         optimizer.zero_grad()

        labels = labels.type(torch.long)
        
        if device is not None:
            idxed_text, labels = idxed_text.to(device), labels.to(device)
            model = model.to(device)
        else:
            idxed_text, labels = idxed_text.cuda(), labels.cuda()
            model = model.cuda()

        predictions = model(idxed_text)
        # predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, labels.type_as(predictions))
        loss.backward()
        
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), -0.5)
        optimizer.step()
        optimizer.zero_grad()

        # prevent internal pytorch timeout due to too many file opens by multiprocessing
        copied_labels = copy.deepcopy(labels.detach().cpu().numpy())
        del labels
        order_labels.extend(copied_labels)

        copied_preds = copy.deepcopy(predictions.detach().cpu().numpy())
        del predictions
        prediction_scores.extend(copied_preds)

        epoch_loss += loss.item()

        if test:
            if counter >= test:
                break
            counter += 1
        
    epoch_metric = roc_auc_score(
        order_labels, torch.sigmoid(torch.Tensor(prediction_scores))
    )

    return epoch_loss / len(dataloader), epoch_metric


def epoch_val_lstm(model, dataloader, criterion, return_preds=False, test=0, device=None):
    """
    Evaluate model on a dataset

    Args:
        model : any pytorch model with defined forward
        dataloader : iterator for dataset, yields (ids, sequence, seq length, labels)
        criterion: loss function
        device: cpu or gpu
        return_preds : bool default False
                       If enabled, returns (ids, predictions, labels, attn, events)

        detach_hidden : bool default False
                        Set to true if AttentionRNN is used
                        Model must have .init_hidden function defined
        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
        max_len (int) : maximum length for attention

    Returns:
        tuple containing:
            average loss for whole epoch,
            average AUC for whole epoch
            if return_preds is enabled, also returns additional tuple:
                ids,
                predictions
                labels
                attn,
                events
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    # initialize lists to compare predictions & ground truth labels for metric calculation
    order_labels = []
    prediction_scores = []

    if test:  # test function on small number of batches
        counter = 0
    with torch.no_grad():

        for idx, (ids, labels, idxed_text) in enumerate(dataloader):

            labels = labels.type(torch.long)
            
            if device is not None:
                idxed_text, labels = idxed_text.to(device), labels.to(device)
            else:
                idxed_text, labels = idxed_text.cuda(), labels.cuda()

            predictions = model(idxed_text)
            #loss = criterion(predictions, labels.squeeze(1))
            loss = criterion(predictions, labels.type_as(predictions))
            epoch_loss += loss.item()

            # prevent internal pytorch timeout due to too many file opens by multiprocessing
            copied_labels = copy.deepcopy(labels.detach().cpu().numpy())
            del labels
            order_labels.extend(copied_labels)

            copied_preds = copy.deepcopy(predictions.detach().cpu().numpy())
            del predictions
            prediction_scores.extend(copied_preds)

            if test:
                if counter >= test:
                    break
                counter += 1

    epoch_metric = roc_auc_score(
        order_labels, torch.sigmoid(torch.Tensor(prediction_scores))
    )
    
    if return_preds:
        return epoch_loss / len(dataloader), epoch_metric, order_labels, torch.sigmoid(torch.Tensor(prediction_scores))

    return epoch_loss / len(dataloader), epoch_metric
import time
import torch
import numbers
from collections import Counter
import torch
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import text_classification
from torchtext.vocab import Vocab

import metrics
import numpy as np

_t0 = time.time()
def log(*args, **kwargs):
    global _t0
    print("{:8.2f}: ".format((time.time() - _t0) / 60), end="")
    print(*args, **kwargs)

def generate_batch(batch):
    '''Reformat data for fn_collate'''
    # batch is a list of tuple (pid, label, text)
    
    # label dims [batch_size]
    y = [entry[1] for entry in batch]
    label = torch.stack(y).float()
    
    # entry is variable length, padded with 0
    # text dims [batch_size, batch_length]
    text = [entry[2] for entry in batch]
    
    
    # used by pytorch to know the actual length of each sequence
    # offsets dims [batch_size]
    offsets = torch.tensor([len(entry) for entry in text], dtype=torch.int64)
    
    text = pad_sequence(text, batch_first=True)
    
    ids = [entry[0] for entry in batch]
    
    return ids, text, offsets, label

def count_parameters(model):
    '''Count total number of parameters to update in model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    '''Calculate total amount fo time spent training'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def build_lstm_dataset(
    datapath,
    min_freq=10,
    uid_colname="discharge_id",
    target_colname="unplanned_readmission",
    x_inputs=[str(x) for x in range(999, -1, -1)],
    target_value="True",
    vocab=None,
    pos_labs_vocab=False,
    nrows=1e9,
    rev=True,
):
    '''
    Reads in dataset line by line, creates vocabulary and subsets dataset to list of arrays to be used later.
    Note: Key difference to transformer build_dataset (has remove_death, build_vocab functions together):
        1. Line by line reading, as the 1000 format CSV file takes a long time to read via pandas
        2. Incorporates remove death immediately
        3. Builds vocabulary here
        
    Arguments:
    ---------
        datapath (str) : path to 1000 CSV file
        min_freq (int) : minimum frequency for vocabulary
        uid_colname (str) : name of unique identifier in each row (patient_id, discharge_dt)
                            returned in list of arrays
        target_colname (str) : column name of target label
        x_inputs (list) : list of columns to be used as inputs
        target_value (str) : value to take as positive label
        vocab (Vocab) : default None creates vocabulary. Will use provided vocab if given.
        pos_labs_vocab (bool) : use only vocabulary from positive label cases
        nrows (int) : number of rows to process
    
    Returns:
    --------
        a list of data and attributes in the following order: 
            patientid_dischargeid key, 
            sequence of events, 
            targets, and 
            mask (identifying padded regions)
    '''
    
    def valid_token(t):
        if len(t) == 0 or t == "<pad>":
            return False
        return True
    
    log("Build token list")
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
            if 'death' in line:
                deaths += 1
                pass
            
            tokens = line.split(",")
            
            if len(tokens[uid_index]) == 0:
                invalid_uid += 1 # some UIDS are missing
                pass
            else:
                uid = tokens[uid_index]
                
            label = 1 if tokens[target_index].startswith(target_value) else 0

            tokens = [tokens[idx] for idx in x_idxes]
            tokens = [t.strip().replace('\n', '') for t in tokens if valid_token(t)]
            
            if rev:
                tokens = tokens[::-1]
            
            token_list.append((uid, label, tokens))
            
            line = f.readline()
            
            if len(token_list) == nrows:
                break
            
    if vocab is None:
        log("Build counter")
        counter = Counter()
        if pos_labs_vocab:
            for uid, label, tokens in token_list:
                if label:
                    counter.update(tokens)
        else:
            for uid, label, tokens in token_list:
                counter.update(tokens)
                
        log("Build vocab")
        vocab = Vocab(counter, min_freq=min_freq, specials=["<pad>", "<unk>"])
        
    log("Build data")
    data = [
        (pid, torch.tensor([label]), torch.tensor([vocab.stoi[t] for t in tokens]))
        for pid, label, tokens in token_list
    ]

    labels = set(["readmission"])
    
    log("Build pytorch dataset")
    log(f"Skipped {invalid_uid} invalid patients")
    log(f"Skipped {deaths} dead patients")
    
    dataset = text_classification.TextClassificationDataset(vocab, data, labels)
    
    log("Done")
    
    return dataset


def epoch_train_lstm(
    model,
    dataloader,
    optimizer,
    criterion,
    test=0
):
    """
    Train model for an epoch, called by ModelProcess function

    detach_hidden is used to detach hidden state between batches,
    and will add a new hidden state to model. Model must have .init_hidden function defined
    
    Args:
    -----
        model (nn.Module): lstm general attention model
        dataloader : iterator for dataset, yields (ids, sequence, seq length, labels)
        criterion : loss function

        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
        
    Returns:
    ----------
        tuple containing:
            average loss for whole epoch,
            average AUC for whole epoch
    """
    import copy
    from sklearn.metrics import roc_auc_score
    
    def repackage_hidden(h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        Needed to prevent RNN+Attention backpropagating between batches.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)
        
    epoch_loss = 0
    epoch_metric = 0

    model.train()

    # initialize lists to compare predictions & ground truth labels for metric calculation
    order_labels = []
    prediction_scores = []
    if test: # test function on small number of batches
        counter = 0
        
    for idx, (ids, text, text_lengths, labels) in enumerate(dataloader):
        
        optimizer.zero_grad()
  
        hidden = model.init_hidden(text.shape[0])
        hidden = repackage_hidden(hidden)

        text, text_lengths, labels = (
            text.to(model.device),
            text_lengths,
            labels.to(model.device),
        )

        predictions, hidden = model(text, text_lengths, hidden)        
        #predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, labels.type_as(predictions))
        loss.backward()
        optimizer.step()

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
    
    epoch_metric = roc_auc_score(order_labels, torch.sigmoid(torch.Tensor(prediction_scores)))
        
    return epoch_loss / len(dataloader), epoch_metric

def epoch_val_lstm(
    model, dataloader, criterion, return_preds=False, test=0, max_len=1000
):
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
    import copy
    from sklearn.metrics import roc_auc_score

    def repackage_hidden(h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        Needed to prevent RNN+Attention backpropagating between batches.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def detach_and_copy(val):
        copied = copy.deepcopy(val.detach().cpu().numpy())
        del val
        return copied

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    # initialize lists to compare predictions & ground truth labels for metric calculation
    order_labels = []
    prediction_scores = []

    if return_preds:
        ids_lst = []
        attn = [None] * len(dataloader)
        feat = [None] * len(dataloader)

    if test:  # test function on small number of batches
        counter = 0
    with torch.no_grad():

        for idx, (ids, text, text_lengths, labels) in enumerate(dataloader):

            text, text_lengths, labels = (
                text.to(model.device),
                text_lengths,
                labels.to(model.device),
            )

            hidden = model.init_hidden(text.shape[0])
            hidden = repackage_hidden(hidden)

            predictions, hidden, attn_weights = model(
                text, text_lengths, hidden, explain=True
            )

            loss = criterion(predictions, labels.type_as(predictions))

            epoch_loss += loss.item()

            # prevent internal pytorch timeout due to too many file opens by multiprocessing
            order_labels.extend(detach_and_copy(labels))
            prediction_scores.extend(detach_and_copy(predictions))

            if return_preds:
                ids_lst.extend(copy.deepcopy(ids))
                attn[idx] = detach_and_copy(attn_weights)
                feat[idx] = detach_and_copy(text)

            epoch_loss += loss.item()

            if test:
                if counter >= test:
                    break
                counter += 1

    epoch_metric = roc_auc_score(order_labels, torch.sigmoid(torch.Tensor(prediction_scores)))

    return_tuple = (epoch_loss / len(dataloader), epoch_metric)
    
    if return_preds:
        #print(len(attn))
        #print(attn)
        # return sizing not always consistent: ensure same dimensions and length
        attn = [
            np.squeeze(cur_attn, 2) if len(cur_attn.shape) == 3 else cur_attn
            for cur_attn in attn
        ]
        
        attn = [
            np.concatenate(  # append with zeros
                (
                    cur_attn,
                    np.zeros(
                        (cur_attn.shape[0], abs(cur_attn.shape[1] - 1000))
                    ),
                ),
                1,
            )
            if cur_attn.shape[1] != 1000
            else cur_attn
            for cur_attn in attn
        ]
        attn = np.concatenate(attn)
            
        feat = [
            np.concatenate(
                (cur_feat, 
                 np.zeros(
                    (cur_feat.shape[0], abs(cur_feat.shape[1] - 1000)))
                ), 1
            )
            if cur_feat.shape[1] != 1000
            else cur_feat
            for cur_feat in feat
            
        ]
        feat = np.concatenate(feat)
        
        return_tuple = return_tuple + ((ids_lst, prediction_scores, order_labels, attn, feat),)

    return return_tuple


def epoch_val_lstm_v2(
    model, dataloader, criterion, return_preds=False, test=0, max_len=1000
):
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
    import copy
    from sklearn.metrics import roc_auc_score

    def repackage_hidden(h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        Needed to prevent RNN+Attention backpropagating between batches.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def detach_and_copy(val):
        copied = copy.deepcopy(val.detach().cpu().numpy())
        del val
        return copied

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    # initialize lists to compare predictions & ground truth labels for metric calculation
    order_labels = []
    prediction_scores = []

    if return_preds:
        ids_lst = []
        attn = [None] * len(dataloader)
        feat = [None] * len(dataloader)

    if test:  # test function on small number of batches
        counter = 0
    with torch.no_grad():

        for idx, (ids, text, text_lengths, labels) in enumerate(dataloader):

            text, text_lengths, labels = (
                text.to(model.device),
                text_lengths,
                labels.to(model.device),
            )

            #hidden = model.init_hidden(text.shape[0])
            #hidden = repackage_hidden(hidden)

            predictions, hidden, attn_weights = model(
                text, text_lengths, text.shape[0], explain=True
            )

            loss = criterion(predictions, labels.type_as(predictions))

            epoch_loss += loss.item()

            # prevent internal pytorch timeout due to too many file opens by multiprocessing
            order_labels.extend(detach_and_copy(labels))
            prediction_scores.extend(detach_and_copy(predictions))

            if return_preds:
                ids_lst.extend(copy.deepcopy(ids))
                attn[idx] = detach_and_copy(attn_weights)
                feat[idx] = detach_and_copy(text)

            epoch_loss += loss.item()

            if test:
                if counter >= test:
                    break
                counter += 1

    epoch_metric = roc_auc_score(order_labels, torch.sigmoid(torch.Tensor(prediction_scores)))

    return_tuple = (epoch_loss / len(dataloader), epoch_metric)
    
    if return_preds:
        #print(len(attn))
        #print(attn)
        # return sizing not always consistent: ensure same dimensions and length
        attn = [
            np.squeeze(cur_attn, 2) if len(cur_attn.shape) == 3 else cur_attn
            for cur_attn in attn
        ]
        
        attn = [
            np.concatenate(  # append with zeros
                (
                    cur_attn,
                    np.zeros(
                        (cur_attn.shape[0], abs(cur_attn.shape[1] - 1000))
                    ),
                ),
                1,
            )
            if cur_attn.shape[1] != 1000
            else cur_attn
            for cur_attn in attn
        ]
        attn = np.concatenate(attn)
            
        feat = [
            np.concatenate(
                (cur_feat, 
                 np.zeros(
                    (cur_feat.shape[0], abs(cur_feat.shape[1] - 1000)))
                ), 1
            )
            if cur_feat.shape[1] != 1000
            else cur_feat
            for cur_feat in feat
            
        ]
        feat = np.concatenate(feat)
        
        return_tuple = return_tuple + ((ids_lst, prediction_scores, order_labels, attn, feat),)

    return return_tuple


def get_average_accuracy(preds, y):
    """
    Returns accuracy, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    Averaged between all the classes, if num_class > 1
    
    Args:
        preds (torch.Tensor): predictions from model
        y (torch.Tensor): correct labels 
    
    Returns:
        float: calculated accuracy
    """
    
    num_class = preds.shape[1] if preds.dim() == 2 else preds.dim()
    acc = ((preds > 0) == (y == 1)).sum().item() / num_class
    acc /= len(preds)

    return acc

def get_individual_acc(preds, y):
    """
    Calculates accuracy for each class and return a list of 
    accuracy for each of the class in corresponding order.
    Assumes multiple classes are available
    
    Args:
        preds (torch.Tensor): predictions from model
        y (torch.Tensor): correct labels 
    
    Returns:
        list: calculated accuracy    
    """
    
    num_class = preds.shape[1]
    acc_lst = [None] * num_class
    for idx in range(num_class):
        acc_lst[idx] = get_average_accuracy(preds[:, idx], y[:, idx])
        
    return acc_lst


def evaluate(model, iterator, criterion, device, return_preds=False, detach_hidden=False, batch_size=0):
    """
    Evaluate model on a dataset
    
    Args:
        model : any pytorch model with defined forward
        iterator : iterator for dataset, yields (sequence, seq length, labels)
        criterion: loss function
        device: cpu or gpu
        return_preds : bool default False
                       If enabled, returns predictions  labels
                       
        detach_hidden : bool default False
                        Set to true if AttentionRNN is used
                        Model must have .init_hidden function defined
        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
                     
    Returns:
        tuple containing:
            average loss for whole epoch,
            average accuracy for whole epoch
            if return_preds is enabled, also returns:
                predictions
                labels
    """    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    if return_preds:
        preds = [None] * len(iterator)
        y = [None] * len(iterator)        
        
    with torch.no_grad():
        for idx, (ids, text, text_lengths, labels) in enumerate(iterator):
            
            text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
            
            if detach_hidden:
                hidden = model.init_hidden(batch_size)
                hidden = repackage_hidden(hidden)
                predictions, hidden = model(text, text_lengths, hidden)
                
            else:
                predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, labels.type_as(predictions))

            acc = get_average_accuracy(predictions, labels)
            
            if return_preds:
                preds[idx] = predictions
                y[idx] = labels
            
            epoch_loss += loss.item()
            epoch_acc += acc
    
    if return_preds:
        return epoch_loss / len(iterator), epoch_acc / len(iterator), torch.cat(preds), torch.cat(y)
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)




def evaluate_explain(model, iterator, criterion, device, return_preds=False, 
                     detach_hidden=False, batch_size=0, explain=False,
                     max_length=1000):
    """
    Evaluate model on a dataset
    
    Args:
        model : any pytorch model with defined forward
        iterator : iterator for dataset, yields (sequence, seq length, labels)
        criterion: loss function
        device: cpu or gpu
        return_preds : bool default False
                       If enabled, returns predictions  labels
                       
        detach_hidden : bool default False
                        Set to true if AttentionRNN is used
                        Model must have .init_hidden function defined
        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
                     
    Returns:
        tuple containing:
            average loss for whole epoch,
            average accuracy for whole epoch
            if return_preds is enabled, also returns:
                predictions
                labels
    """    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    if return_preds:
        preds = [None] * len(iterator)
        y = [None] * len(iterator) 
    
    if explain:
        attn = [None] * len(iterator)
        feat = [None] * len(iterator)
         
    with torch.no_grad():
        for idx, (text, text_lengths, labels) in enumerate(iterator):
            
            #if text.size()[1] < 1000:
            #    delta_text = abs(text.size()[1]-1000)
            #    text = torch.cat((text, torch.zeros((text.size()[0], delta_text), dtype=text.dtype)), 1)
            #    print(text.size())
            text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
            
            if detach_hidden:
                hidden = model.init_hidden(batch_size)
                hidden = repackage_hidden(hidden)
                
                if explain:
                    predictions, hidden, attn_weights = model(text, text_lengths, hidden, explain=True)
                else:
                    predictions, hidden = model(text, text_lengths, hidden)
                
            else:
                predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, labels.type_as(predictions))

            acc = get_average_accuracy(predictions, labels)
            
            if return_preds:
                preds[idx] = predictions
                y[idx] = labels
            
            if explain:
                attn[idx] = attn_weights
                feat[idx] = text
                
            epoch_loss += loss.item()
            epoch_acc += acc
    
    return_tuple = (epoch_loss / len(iterator), epoch_acc / len(iterator))
    
    
    if return_preds:
        return_tuple = return_tuple + (torch.cat(preds), torch.cat(y),)
        
    if explain:
        for idx, cur_attn in enumerate(attn):
            if len(cur_attn.size())== 3:
                cur_attn = cur_attn.squeeze(2)
                attn[idx]= cur_attn
            if cur_attn.size()[1] != max_length:
                
                delta_text = abs(cur_attn.size()[1]-max_length)
                cur_attn = cur_attn.cpu()
                attn[idx] = torch.cat((cur_attn, torch.zeros((cur_attn.size()[0], delta_text), dtype=cur_attn.dtype)), 1).to(device)
       
        for idx, cur_feat in enumerate(feat):
            if cur_feat.size()[1] != max_length:
                delta_text = abs(cur_feat.size()[1]-max_length)
                cur_feat = cur_feat.cpu()
                #print(cur_feat.size())
                feat[idx] = torch.cat((cur_feat, torch.zeros((cur_feat.size()[0], delta_text), dtype=cur_feat.dtype)), 1).to(device)                
            
        
        return_tuple = return_tuple + (torch.cat(attn), torch.cat(feat), )
    
    #print(len(return_tuple))
    return return_tuple



def train_pos_neg(model, pos_iterator, neg_iterator, optimizer, criterion, device, 
                  detach_hidden=False, batch_size=0,
                  metric='acc'):
    """
    Train model for an epoch.
    
    Args:
        model (nn.Module): any pytorch model with defined forward
        iterator : iterator for dataset, yields (sequence, seq length, labels)
        criterion : loss function
        device : cpu or gpu
        detach_hidden : bool default False
                        Used to detach hidden state between batches,
                        and will add a new hidden state to model
                        Model must have .init_hidden function defined
        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
        
    Returns:
        tuple containing:
            average loss for whole epoch,
            average accuracy for whole epoch
    """
    epoch_loss = 0
    epoch_metric = 0
    
    model.train()
    
    for idx, ((pos_text, pos_text_lengths, pos_labels), (neg_text, neg_text_lengths, neg_labels)) in enumerate(zip(pos_iterator, neg_iterator)):

        text = torch.cat((pos_text, neg_text))
        text_lengths = torch.cat((pos_text_lengths, neg_text_lengths))
        labels = torch.cat((pos_labels, neg_labels))

        optimizer.zero_grad()

        if detach_hidden:
            if batch_size == 0:
                raise ValueError('Batch_size in training needs to number with detach_hidden')
            hidden = model.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)

        text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)

        if detach_hidden:
            predictions, hidden = model(text, text_lengths, hidden)
        else:
            predictions = model(text, text_lengths).squeeze(1)


        loss = criterion(predictions, labels.type_as(predictions))
        loss.backward()

        optimizer.step()

        if metric == 'acc':
            batch_metric = get_average_accuracy(predictions, labels)
        else:
            batch_metric = metrics.compute_single_metric(labels.cpu().numpy(), predictions.detach().cpu().numpy())

        epoch_loss += loss.item()
        epoch_metric += batch_metric

    
    return epoch_loss / len(iterator), epoch_metric / len(iterator)




##### OLD CODE
def build_text_dataset(df, feat_colnames, label_colnames, vocab=None, min_freq=1):
    """
    Build pytorch text classification dataset.
    
    Args:
        df (dataframe): contains both features and labels
        feat_colnames (list): columns that the sequence resides in
        label_colnames (list): columns that contains labeling information
        vocab (Vocab): vocabulary to be used with this dataset.
                       default None, which function will build vocabulary
                       based on words in df[feat_colnames]
        min_freq (int): minimum frequency to use for building vocabulary
                        only used if need to build vocabulary
                        
    Returns:
        (TextClassificationDataset)
    
    """
    
    def create_tuple(labels, features, vocab):
        try:
            sentence = [x for x in features if str(x).lower() != 'nan']
        
            sentence = torch.tensor([vocab.stoi[x] if x in vocab.stoi.keys() else vocab.stoi['<unk>'] for x in sentence])
        
            return (torch.tensor(labels), sentence)
        except Exception as excpt:
            print(sentence)
            print(excpt)
            raise ValueError
        else:
            return vocab

    if vocab is None:
        counter = Counter()
        words = df[feat_colnames].values.ravel('K')
        words = [str(x) for x in words if str(x).lower() != 'nan']
        
        
        counter.update(words)
        if not isinstance(min_freq, numbers.Number):
            raise ValueError(f'Something wrong with {min_freq}')
            
        
        vocab = Vocab(counter, min_freq=min_freq, specials=['<pad>', '<unk>'])
        
        print('Completed vocabulary')
    else:
        print('Vocab already supplied')
    
    # create dataset
    data = df.apply(
        lambda row: create_tuple(row[label_colnames],
                                 row[feat_colnames],
                                 vocab), axis=1)
    
    labels = set(df[label_colnames])
    
    new_dataset = text_classification.TextClassificationDataset(vocab, data, labels)
    print('New dataset created')
    
    return new_dataset


def train(model, iterator, optimizer, criterion, device, detach_hidden=False, batch_size=0,
          metric='acc'):
    """
    Train model for an epoch.
    
    Args:
        model (nn.Module): any pytorch model with defined forward
        iterator : iterator for dataset, yields (sequence, seq length, labels)
        criterion : loss function
        device : cpu or gpu
        detach_hidden : bool default False
                        Used to detach hidden state between batches,
                        and will add a new hidden state to model
                        Model must have .init_hidden function defined
        batch_size : int default 0
                     used when detach_hidden is enabled
                     to create the correct hidden sizes during initialization
        
    Returns:
        tuple containing:
            average loss for whole epoch,
            average accuracy for whole epoch
    """
    epoch_loss = 0
    epoch_metric = 0
    
    model.train()
    
    for idx, (ids, text, text_lengths, labels) in enumerate(iterator):
        
        optimizer.zero_grad()
        
        if detach_hidden:
            if batch_size == 0:
                raise ValueError('Batch_size in training needs to number with detach_hidden')
            hidden = model.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)
        
        text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
        
        if detach_hidden:
            predictions, hidden = model(text, text_lengths, hidden)
        else:
            predictions = model(text, text_lengths).squeeze(1)

        
        loss = criterion(predictions, labels.type_as(predictions))
        loss.backward()
        
        optimizer.step()
        
        if metric == 'acc':
            batch_metric = get_average_accuracy(predictions, labels)
        else:
            batch_metric = metrics.compute_single_metric(labels.cpu().numpy(), predictions.detach().cpu().numpy())
        
        epoch_loss += loss.item()
        epoch_metric += batch_metric

    return epoch_loss / len(iterator), epoch_metric / len(iterator)



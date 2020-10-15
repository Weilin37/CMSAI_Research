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

def generate_batch(batch):
    '''Reformat data for fn_collate'''
    # batch is a list of tuple (pid, label, text)
    
    # label dims [batch_size]
    y = [entry[1] for entry in batch]
    label = torch.stack(y).float()
    #label = torch.tensor([entry[0] for entry in batch])
    
    # entry is variable length, padded with 0
    # text dims [batch_size, batch_length]
    text = [entry[2] for entry in batch]
    text = pad_sequence(text, batch_first=True)
    
    # used by pytorch to know the actual length of each sequence
    # offsets dims [batch_size]
    offsets = torch.tensor([len(entry) for entry in text])
    
    return text, offsets, label

def count_parameters(model):
    '''Count total number of parameters to update in model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    '''Calculate total amount fo time spent training'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

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

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    Needed to prevent RNN+Attention backpropagating between batches.
    """

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

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
        for idx, (text, text_lengths, labels) in enumerate(iterator):
            
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
    
    for idx, (text, text_lengths, labels) in enumerate(iterator):
        
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

def evaluate_explain(model, iterator, criterion, device, return_preds=False, detach_hidden=False, batch_size=0,
                     explain=False):
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
        return_tuple = return_tuple + (torch.cat(attn), torch.cat(feat), )
    
    return return_tuple
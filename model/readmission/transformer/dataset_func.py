import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

def build_dataset(
    input_df, vocab, feat_colnames, label_colnames, day_length=30, max_length=50
):
    """
    Subsets the entire dataset into a list of arrays to be used later.
    - Specific vocabulary
    - By number of days (whole dataset is 365)
    
    Arguments:
    ----------
        input_df (dataframe) : input features and labels in dataframe style
        vocab (Vocab) : vocabulary to be used to one-hot encode
        feat_colnames (list) : list of columns to be used in correct sequence for features
        label_colnames (list) : target label 
        day_length (int) : sequence length of feat_colnames to use
        max_length (int) : maximum sequence length
        
    Returns:
    --------
        a list of data and attributes in the following order: 
            patientid_dischargeid key, 
            sequence of events, 
            targets, and 
            mask (identifying padded regions)
    """
    # create dataset
    print("=" * 20 + "New dataset created" + "=" * 20 + "\n")
    print("Used days: ", feat_colnames[-day_length], feat_colnames[-1])

    data = input_df[feat_colnames[-day_length:]].to_numpy()

    sequence = []
    pad_mask = []
    print("Total size before building dataset: ", data.shape)

    # convert all events into indexes from vocab
    for i in range(len(data)):
        sentence = []
        mask = []

        for j in range(len(data[i])):
            words = str(data[i][j]).replace("d_s", "d_").replace(' ', '').split(",")
            words = sorted(
                [
                    vocab.stoi[w] if w in vocab.stoi else vocab.stoi["<unk>"]
                    for w in words
                ]
            )

            if len(words) > max_length:
                words = words[:max_length]

            words = words + [vocab.stoi["<pad>"]] * (max_length - len(words))
            sentence.append(words)
            mask.append([0.0 if w == vocab.stoi["<pad>"] else 1.0 for w in words])

        sequence.append(sentence)
        pad_mask.append(mask)

    
    print("New dataset created")
    print("Sequence length: ", len(sequence))

    discharge_ids = np.array(input_df["discharge_id"])
    labels = np.array(input_df[label_colnames])
    sequence = np.array(sequence)
    pad_mask = np.array(pad_mask)
    
    return [discharge_ids, sequence, labels, pad_mask]
    
from torch.utils.data import Dataset, DataLoader
import torch


class BuildDataset(Dataset):
    """
    Create Pytorch Dataset class to be used by data loader. Extract input, labels, mask from 
    an existing Python list
    
    Arguments:
    -----------
        seq_length (int) : number of days as input features
        event_length (int) : number of events in a single day to be extracted
        data_list (list) : list of input data in the order of (uids, input_features, labels, mask)
        
    Returns:
    ---------
        Dataset
    """

    def __init__(self, seq_length=30, event_length=30, data_list=None):
        """mode: 'read' will process data"""
        self.uids, self.data, self.label, self.mask = self.ProcessData(
            data_list, seq_length, event_length
        )

    def ProcessData(self, data_list, seq_length, event_length):
        uids, input_data, labels, mask = (
            data_list[0],
            data_list[1][:, -seq_length:, :event_length],
            data_list[2],
            data_list[3][:, -seq_length:, :event_length],
        )
        return uids, input_data, labels, mask

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.uids[idx],
            torch.tensor(self.data[idx]),
            torch.tensor(self.label[idx]),
            torch.tensor(self.mask[idx]),
        )

def get_dataloader(input_dataset, batch_size=64, shuffle=True, num_workers=4):
    '''
    Return Pytorch DataLoader
    '''
    return DataLoader(
        input_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return [discharge_ids, sequence, labels, pad_mask]

def read_pickled_ds(file_dir, seq_length, event_length):
    '''
    Function to read is specific number of recent events from specific number of days
    
    Used to read a large dataset, and script will apply kfold instead.
    '''
    with open(file_dir, 'rb') as f:
        ids, data, label, mask = pickle.load(f)
        ids = ids.astype(str)
        cut_data = data[:,-seq_length:,:event_length]
        cut_mask = mask[:,-seq_length:,:event_length]
        label = label.astype(int)
        
    return ids, cut_data, label, mask
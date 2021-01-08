import pandas as pd
import numpy as np
import os
import pickle
import numbers
from collections import Counter
import torch
from torchtext.vocab import Vocab

def read_data(
    data_fp, drop_duplicates=False, check=True, y_target=None, uid=None, test=0
):
    """
    Function to read-in, check and process raw 365 training or test data
    
    Arguments:
    ----------
        data_fp (str) : filepath to csv
        drop_duplicates (bool) : drop duplicate UID (must provide uid)
        check (bool) : check for data quality (nrows, duplicates, label ratio)
                       must provide uid, y_target
        y_target (str) : default None
                         column name of label
        uid (str) : default None
                    column name of UID of dataset (usually patient_id + discharge_id)
        test (int) : default 0
                     if not zero, will read in test number of rows
                    
    Returns:
    --------
        data_df (dataframe) : cleaned dataframe
    
    """
    if not os.path.isfile(data_fp):
        raise Exception(f"Invalid data filepath: {data_fp}")

    if test:
        data_df = pd.read_csv(data_fp, low_memory=False, nrows=test)
    else:
        data_df = pd.read_csv(data_fp, low_memory=False)
    print(f"Read data from {data_fp}\n")

    num_dropped = 0
    if drop_duplicates:
        if uid not in data_df.columns:
            raise Exception("Missing UID, unable to drop duplicates")

        num_droppped = data_df.shape[0]
        data_df.drop_duplicates(uid, inplace=True)
        num_dropped -= data_df.shape[0]

    if check:
        print("=" * 20 + "Checking data" + "=" * 20 + "\n")
        print(f"Data size: {data_df.shape}\n")

        if drop_duplicates:
            print(f"Number of duplicate rows: {num_dropped}\n")

        if (y_target is not None) and (y_target in data_df.columns):
            print(f"Label ratio for {y_target}")
            print(data_df[y_target].value_counts(normalize=True))

        if (uid is not None) and (uid in data_df.columns):
            nb_duplicates = data_df[uid].duplicated().sum()
            print(f"\n{uid} duplicates: {nb_duplicates}")

    return data_df


def remove_death(data_df, y_target, x_inputs, bad_word="death"):
    """
    Removes any rows in data that contains unwanted words, i.e. death
    in any of the x_input columns
    
    Arguments:
    ----------
        data_df (dataframe) : data to process
        x_inputs (list) : list of input columns to consider when removing words
        y_target (str) : column name of label 
        bad_word (str) : word used to decide which rows to remove
        
    Returns:
    --------
        data_df_str (dataframe) : data with rows containing unwanted words removed
                    
    
    """
    data_df_str = data_df.astype(str)
    data_df_str[y_target] = data_df[y_target].astype(int).tolist()[:]

    indices = set()
    for input_col in x_inputs:
        indices.update(data_df_str[data_df_str[input_col].str.contains(bad_word)].index)

    print("\n" + "=" * 20 + "Removing bad word data" + "=" * 20 + "\n")
    print(f"Removing bad words: {len(indices)} rows contain the word {bad_word}")

    return data_df[~data_df.index.isin(indices)]

def build_vocab(
    data_df,
    feat_colnames,
    y_target="unplanned_readmission",
    min_freq=1,
    specials=["<pad>", "<unk>"],
    pos_labs_vocab=True,
):
    """
    Create a vocabulary: This maps all events to an index, including 
    <pad> : index 0, sentence padding
    <unk> : index 1, unknown events
    nan : index 2, no events
    
    Arguments:
    ----------
        data_df (dataframe) : containing features & target
        feat_colnames (list) : input column names to build vocab
        y_target (str) : target column name
        min_freq (int) : minimum frequency for vocab
        specials (list) : special characters (padding, unknown)
        pos_labs_vocab (bool) : to use only words from minority/pos class
        
    Returns:
    --------
        vocab (Vocab) 
    """

    def build_counter(data_df, feat_colnames):
        counter = Counter()
        words = data_df[feat_colnames].values.ravel("K")
        print("start word number: ", words.shape)

        new_words = []

        for x in words:
            x = str(x)
            x = x.replace("d_s", "d_")
            new_words.extend(x.replace(" ", "").split(","))

        print("exact word number: ", len(new_words))

        counter.update(new_words)

        if not isinstance(min_freq, numbers.Number):
            raise ValueError(f"Something wrong with {min_freq}")

        return counter

    print("\n" + "=" * 20 + "Build vocabulary" + "=" * 20 + "\n")
    vocab_df = data_df
    if pos_labs_vocab:
        vocab_df = data_df[data_df[y_target] == True]

    counter = build_counter(vocab_df, feat_colnames)

    vocab = Vocab(counter, min_freq=min_freq, specials=specials, specials_first=True)

    print(f"Completed vocabulary: {len(vocab)} vocabs")

    return vocab
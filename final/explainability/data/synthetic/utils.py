"""Utils module for the generation of synthetic dataset."""

import yaml
import random
import string
import os
import numpy as np
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_tokens(tokens_fp):
    """
    Load tokens from yaml file path
    Args:
        token_fp(str): Tokens path
    Returns:
        Dictionary of tokens
    """
    with open(tokens_fp, "r") as tokens:
        tokens = yaml.safe_load(tokens)
    return tokens


def get_uid(uid_len):
    """
    Get random UID string with a length of uid_len.
    Args:
        uid_len(int): Length of the random UID string
    Returns:
        Randomly generated string as UID
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=uid_len))

def get_a_sequence(seq_len, tokens, token_mappings, seq_type):
    """
    Generate sequence of events for a single patient.
    Args:
        seq_len(int): Length of the sequence of events
        tokens(dict): Dictionary of tokens grouped with their types(adverse/helper/unhelper/noise)
        token_mappings(dict): Mapping of token suffix with their type
        seq_type(str): Sequence type (AAH/UH...)
    Returns:
        List of sequence of events with paddings if needed
    """
    n_seq_type = len(seq_type)
    n_noise = (
        np.max(
            (
                10,
                random.choices(range(n_seq_type, seq_len), k=1)[0],
            )
        )
        - (n_seq_type)
    )
    sel_positions = sorted(random.sample(range(n_noise), k=n_seq_type))
    sel_tokens = []
    for key in seq_type:
        key_mapping = token_mappings[key]
        sel_tokens.append(random.choices(tokens[key_mapping])[0])

    # Randomize sequence
    random.shuffle(sel_tokens)

    sel_tokens = list(zip(sel_positions, sel_tokens))
    sel_noise = get_tokens(seq_len, tokens, "noise_tokens", n_noise)

    for idx, event in sel_tokens:
        sel_noise.insert(idx, event)

    sequence = ["<pad>"] * (seq_len - len(sel_noise)) + sel_noise

    return sequence


def get_sequences(
    seq_len,
    tokens,
    token_mappings,
    seq_types,
    n_seq_per_type,
    uid_colname,
    uid_len,
):
    """
    Get multiple sequences.
    Args:
        seq_len(int): Length of the sequence of events
        tokens(dict): Dictionary of tokens grouped with their types
        token_mappings(dict): Mapping of token suffix with their type
        seq_types(list): List of all posible sequence types
        n_seq_per_type(int): Number of sequences/rows per sequence type
        uid_colname(str): Name of unique id column name (patient id)
        uid_len(int): Length of the generated patient id
    Returns:
        Sequences of events in dataframe format
    """

    sequences = []
    for seq_type in seq_types:
        for _ in range(n_seq_per_type):
            sequence = get_a_sequence(seq_len, tokens, token_mappings, seq_type)
            uid = get_uid(uid_len)
            sequence = [uid] + sequence + [seq_type]
            sequences.append(sequence)
    random.shuffle(sequences)

    columns = [uid_colname] + [str(x) for x in range(seq_len - 1, -1, -1)] + ['sequence_type']
    df = pd.DataFrame(sequences, columns=columns)
    df.set_index(uid_colname, inplace=True)
    return df


def generate_dataset(df_sequences,
                     seq_len,
                     seq_types,
                     seq_base_probs,
                     data_type
                    ):
    """
    Generate synthetic dataset (event/sequence based).
        seq_len(int): Length of the sequence of events
        tokens(dict): Dictionary of tokens grouped with their types
        token_mappings(dict): Mapping of token suffix with their type
        seq_types(list): List of all posible sequence types
        seq_base_probs(list): List of base probabilities of being +ve event for each seq type
        n_seq_per_type(int): Number of sequences/rows per sequence type
        uid_colname(str): Name of unique id column name (patient id)
        uid_len(int): Length of the generated patient id
        data_type(str): Dataset type (event/sequence)
    Returns:
        Sequences of events and labels in dataframe format
    """
    def get_indices(seq0):
        """
        Get location/index of each non-noise taken in the given sequence of events.
        Args:
            seq0(list): List of events/tokens
        """
        seq = seq0[:]
        seq.reverse()
        idx_seq = {}
        for idx, token in enumerate(seq):
            if (not token.endswith('_N')) and token != '<pad>':
                idx_seq[idx] = token
        return idx_seq

    def get_patient_proba(row, seq_len, seq_types, base_probs, data_type):
        """
        Get probability of a positive labels for a given sequence of events.
        Args:
            row(pandas.Series): Sequence of events for a given patient
            seq_len(int): Sequence length
            seq_types(str): Sequence category(AAH/AH, etc.)
            base_probs(list): Base probability for each category(sequence type)
            data_type(str): Dataset type (event/sequence)
        Returns:
            List of the output probability and label
        """
        feature_names = [str(i) for i in range(seq_len-1, -1, -1)]
        seq_type = row['sequence_type']
        idx = seq_types.index(seq_type)
        base_proba = base_probs[idx]
        seq = row[feature_names].tolist()
        seq = get_indices(seq)
        proba = get_proba(seq, seq_len, base_proba, data_type)
        label = get_label(proba, target=1)
        return [proba, label]

    df = df_sequences.copy()

    results = df.apply(get_patient_proba, axis=1, args=(seq_len, seq_types, seq_base_probs, data_type))
    probs = [results.iloc[i][0] for i in range(len(results))]
    labels = [results.iloc[i][1] for i in range(len(results))]
    df['proba'] = probs
    df['label'] = labels

    return df


def get_proba(seq, seq_len, base_proba, data_type='event'):
    """
    Compute the sequence's probability of being a positive label.
    Args:
        seq(dict): Dictionary of indices/posititions of each non-noise token
        seq_len(int): Sequence length
        base_proba(float): Base probability
        data_type(str): Dataset type (event/sequence)
    Returns:
        Output Probability    
    """
    #If event-based...
    if data_type == 'event':
        return base_proba
    
    #If sequence-based...
    if seq_len == 30:
        multiplier = 0.5
    else:
        multiplier = 0.05
        
    a = 0.2  # Constant for Adverse
    h = 0.4  # Constant for helper
    u = 0.75  # Constant for unhelper

    prob = 0
    for ts in seq:
        event = seq[ts]
        if event.endswith('_A'): 
            prob += math.exp(-(a * int(ts) * multiplier)) 
        elif event.endswith('_H'):
            prob += math.exp(-(h * int(ts) * multiplier))
        elif event.endswith('_U'):
            prob -= math.exp(-(u * int(ts) * multiplier))
    prob = max(0.1,min(1, prob))
    return round(prob, 4)


def get_tokens(seq_len, token_dict, token_key, n_tokens):
    """
    Get random list of tokens.
    Args:
        seq_len(int): Sequence length
        token_dict(dict): Dictionary of tokens for each type
        token_key(str): Token type (adverse/helper, etc.)
        n_tokens(int): Number of available tokens
    """
    return random.choices(token_dict[token_key], k=n_tokens)


def get_label(prob_label, target):
    """
    Get random target label.
    Args:
        prob_label(float): Probability of being a positive label
        target(int): Target Label
    Returns:
        Target label
    """
    return target if random.random() <= prob_label else 1 - target


def plot_features_distribution(df, seq_len=30, split="train", normalize=True):
    """
    Plots distribution of features.
    Args:
        df(pd.DataFrame): Data in pandas dataframe
        seq_len(int): Sequence length
        split(str): Dataset Split
        normalize(bool): Whether to normalize the distribution
    """
    feature_names = [str(i) for i in range(seq_len - 1, -1, -1)]
    X = df[feature_names]
    freqs = dict(Counter(X.values.flatten()))
    del freqs["<pad>"]

    if normalize:
        total = sum(freqs.values(), 0.0)
        for key in freqs:
            freqs[key] = freqs[key] / total * 100.0

    freqs = dict(sorted(freqs.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=list(range(len(list(freqs.keys())))),
        y=list(freqs.values()),
        orient="v",
    )
    z = ax.set_xticklabels(list(freqs.keys()), rotation=90)

    y_label = "Frequency"
    if normalize:
        y_label += "(%)"
    plt.title(f"Features Distribution ({split} set)")
    plt.xlabel("Features")
    plt.ylabel(y_label)
    plt.show()
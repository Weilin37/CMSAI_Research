"""Utils Module for the creation of the toy dataset."""

import yaml
import pandas as pd
import random
import string
import os


def load_tokens(tokens_fp):
    """Load tokens from yaml file path"""
    with open(tokens_fp, "r") as tokens:
        tokens = yaml.safe_load(tokens)
    return tokens


def get_uid(uid_len):
    """Get random UID string with a length of uid_len."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=uid_len))


def get_idx_tok(seq_len, token_dict, token_key, n_pairs):
    """Get random index and token from token_key of n_pairs."""
    return [
        (
            random.choices(range(seq_len), k=1)[0],
            random.choices(token_dict[token_key], k=1)[0],
        )
        for _ in range(n_pairs)
    ]


def get_tokens(seq_len, token_dict, token_key, n_tokens):
    """Get random list of tokens."""
    return random.choices(token_dict[token_key], k=n_tokens)


def get_label(prob_label, target):
    """Get random target label."""
    return target if random.random() <= prob_label else 1 - target


def save_csv(df, fp):
    """Saves data into a csv file."""
    if not os.path.isdir(os.path.split(fp)[0]):
        os.makedirs(os.path.split(fp)[0])

    df.to_csv(fp, index=False)


# Sequence generation functions:
# - get_a_sequence
# - get_sequences
def get_a_sequence(adverse, helper, unhelper, seq_len, label, tokens):
    """creates sequence + label (at the end of list). returns list of list"""
    n_noise = random.choices(range(adverse + helper + unhelper + 1, seq_len), k=1)[
        0
    ] - (adverse + helper + unhelper)

    sel_adverse, sel_helper, sel_unhelper = [], [], []

    if adverse:
        sel_adverse = get_idx_tok(seq_len, tokens, "adverse_tokens", adverse)
    if helper:
        sel_helper = get_idx_tok(seq_len, tokens, "adverse_helper_tokens", helper)
    if unhelper:
        sel_unhelper = get_idx_tok(seq_len, tokens, "adverse_unhelper_tokens", unhelper)

    sel_noise = get_tokens(seq_len, tokens, "noise_tokens", n_noise)

    for idx, event in sel_adverse + sel_helper + sel_unhelper:
        sel_noise.insert(idx, event)

    sel_noise = ["<pad>"] * (seq_len - len(sel_noise)) + sel_noise

    sim_lab = get_label(0.9, target=label)
    return sel_noise + [sim_lab]


def get_sequences(
    adverse, helper, unhelper, seq_len, label, uid_len, uid_colname, n_seq, tokens
):
    """Get multiple sequences."""
    sequences = [
        get_a_sequence(
            adverse=adverse,
            helper=helper,
            unhelper=unhelper,
            seq_len=seq_len,
            label=label,
            tokens=tokens,
        )
        + [get_uid(uid_len)]
        for _ in range(n_seq)
    ]

    seq_df = pd.DataFrame(sequences)
    seq_df.columns = [str(x) for x in range(seq_len - 1, -1, -1)] + [
        "label",
        uid_colname,
    ]

    return seq_df


def get_simple_dataset(seq_len, uid_len, uid_colname, count_dict, tokens):
    """Get a simple toy dataset."""
    ppp = get_sequences(
        adverse=1,
        helper=1,
        unhelper=0,
        seq_len=seq_len,
        label=1,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_ppp_adverse"],
        tokens=tokens,
    )
    pp = get_sequences(
        adverse=1,
        helper=0,
        unhelper=0,
        seq_len=seq_len,
        label=1,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_pp_adverse"],
        tokens=tokens,
    )
    p = get_sequences(
        adverse=0,
        helper=3,
        unhelper=0,
        seq_len=seq_len,
        label=1,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_p_adverse"],
        tokens=tokens,
    )
    nnn = get_sequences(
        adverse=0,
        helper=0,
        unhelper=3,
        seq_len=seq_len,
        label=0,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_nnn_adverse"],
        tokens=tokens,
    )
    nn = get_sequences(
        adverse=0,
        helper=1,
        unhelper=2,
        seq_len=seq_len,
        label=0,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_nn_adverse"],
        tokens=tokens,
    )
    n = get_sequences(
        adverse=0,
        helper=2,
        unhelper=1,
        seq_len=seq_len,
        label=0,
        uid_len=uid_len,
        uid_colname=uid_colname,
        n_seq=count_dict["n_n_adverse"],
        tokens=tokens,
    )

    dataset = pd.concat([ppp, pp, p, n, nn, nnn], axis=0)
    dataset.reset_index(inplace=True)
    indexes = [idx for idx in range(dataset.shape[0])]
    random.shuffle(indexes)
    dataset = dataset.iloc[indexes, :]

    print(f"dataset: {dataset.shape}")
    print(f"ratio:\n{dataset.label.value_counts(normalize=True)}\n")

    return dataset

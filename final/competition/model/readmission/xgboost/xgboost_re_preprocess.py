"""
A module that preprocesses readmissions data to be ready for xgboost training.
"""

import json
import os
import time
import numpy as np
import pandas as pd

# import modin.pandas as pd
import torch
import torchtext
from collections import OrderedDict


def get_frequent_features(vocab, num_features, codes_only=True, exclusion_list=[]):
    """
    Get the most frequent codes/features.
    Args:
        vocab(Object): Vocab object
        num_features(int): # of features
        codes_only(bool): Wether to use ICD codes as features
        exclution_list(list): List of events/keys to excluded from features
    Returns:
        List of most frequent features
    """
    num_exc = len(exclusion_list) + 100
    features = vocab.freqs.most_common(num_features + num_exc)
    if codes_only:
        features = [
            word[0]
            for word in features
            if word[0] not in exclusion_list and ("_" in word[0])
        ]
    else:
        features = [word[0] for word in features if word[0] not in exclusion_list]
    features = [word for word in features if "day" not in word]  # Exclude day features
    features = features[:num_features]
    return features


def get_one_hot_frequent_features(row, frequent_features):
    """Gets one-hot encoding of the most frequent features of a given patient data
    Args:
        row(pd.Series): row to specify patient's specific adverse event
    Returns:
        Returns 0 if max value is 0 otherwise 1
    """
    features = set(row.tolist())
    one_hot = [int(ft in features) for ft in frequent_features]
    return one_hot


def read_labels(labels_path):
    """Read list of labels from path.
    Args:
        labels_path(str): Classes path
    Returns:
        List of classes
    """
    with open(labels_path, "r") as fp:
        labels = fp.readlines()
        labels = [label.strip() for label in labels]
    return labels


def get_class_imbalance(df_y):
    """Get class imbalance for all the target variables.
    Args:
        df_y(DataFrame): Dataframe of class imbalances
    Returns:
        Dictionary of class imbalances
    """
    imbalance = df_y.apply(lambda x: x.value_counts()).transpose().values.tolist()
    imbalance = dict(zip(df_y.columns.tolist(), imbalance))
    return imbalance


def preprocess(
    df, features, label, fold, split, output_dir, class_imbalance_fname=None
):
    """Transform the predictor data to one-hot encoding and aggregate with target data.
    Args:
        df(DataFrame): Data to be preprocessed
        features(list): List of features
        label(str): Class/label name
        split(str): Dataset split
        output_dir(str): Output directory
        class_imbalance_fname(str): Class imbalance filename
    Returns:
        Preprocessed data in dataframe
    """
    print("Preprocessing and saving fold={} and split={} data...".format(fold, split))
    df = df[df[label].notna()]

    df_x = df.iloc[:, :1000]
    df_y = df[[label]]
    df_y = df_y.astype(int)

    df_x = df_x.apply(get_one_hot_frequent_features, axis=1, args=(features,))
    df_x = pd.DataFrame(df_x.tolist(), columns=features)
    df = pd.concat([df_x, df_y], axis=1)

    my_output_dir = os.path.join(output_dir, fold)
    if not os.path.exists(my_output_dir):
        os.makedirs(my_output_dir)

    if split == "train":
        imb = get_class_imbalance(df_y)
        class_imbalance_path = os.path.join(my_output_dir, class_imbalance_fname)
        with open(class_imbalance_path, "w") as fp:
            json.dump(imb, fp)

    output_path = os.path.join(my_output_dir, split + ".csv")
    df.to_csv(output_path, index=False)
    print("{} data successfully preprocessed!".format(split))
    return df


def prepare(df, num_features_list, label, output_dir, fold, split="train"):
    """Prepares data for model training.
    Args:
        df(DataFrame): Data to be ready for training
        num_features_list(list): List of number of features
        label(str): Class/label name
        output_dir(str): Output directory
        fold: Data fold number
        split(str): Dataset split
    Returns:
        None
    """
    num_targets = 1
    features = df.columns.tolist()[:-num_targets]
    for num_features in num_features_list:
        print("Preparing data with {} features...".format(num_features))
        columns = [label] + features[:num_features]
        my_output_dir = os.path.join(output_dir, fold, str(num_features))

        if not os.path.exists(my_output_dir):
            os.makedirs(my_output_dir)

        output_path = os.path.join(my_output_dir, split + ".csv")
        df[columns].to_csv(output_path, index=False, header=None)
    print("Successfully prepared data for training!")


def get_all_data_from_folds(data_root_dir, folds, data_all):
    """
    Integrate all the data from folds to be used for final training.
    Args:
        data_root_dir(str): Root data dir
        folds(list): List of data folds
        data_all(str): Name of all data folder
    Returns:
        All data in dataframe format
    """
    output_dir = os.path.join(data_root_dir, data_all)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_all = None
    for fold in folds:
        data_path = os.path.join(
            data_root_dir, fold, "test", "raw_test_data_1000_30days_anony.csv"
        )
        df = pd.read_csv(data_path)
        if df_all is None:
            df_all = df.copy(deep=True)
        else:
            df_all = pd.concat([df_all, df], ignore_index=True, axis=0)
    output_path = os.path.join(output_dir, "train.csv")
    df_all.to_csv(output_path, index=False)
    return df_all


def combine_all_vocabs(
    data_dir, folds, data_all="all", vocab_fname="vocab_1000_vall_30days"
):
    """
    Combine all vocabularies and save to disk.
    Args:
        data_dir(str): Data directory
        folds(list): List of dataset folds
        data_all(str): All data folder
        vocab_fname(str): Vocab filename
    Returns:
        Combined vocab output file path
    """

    def combine_vocabs(vocab1, vocab2):
        """Combine two vocabularies."""
        freqs = vocab1.freqs + vocab2.freqs
        vocab = torchtext.vocab.Vocab(freqs)
        return vocab

    output_dir = os.path.join(data_dir, data_all, "vocab")
    output_path = os.path.join(output_dir, vocab_fname)
    if os.path.exists(output_path):
        print("Aggregated Vocab already saved to {}!".format(output_path))
        return output_path

    print("Aggregating vocabularies...")
    vocab_all = None
    for fold in folds:
        vocab_path = os.path.join(data_dir, fold, "vocab", vocab_fname)
        if vocab_all is None:
            vocab_all = torch.load(vocab_path)
        else:
            vocab = torch.load(vocab_path)
            vocab_all = combine_vocabs(vocab_all, vocab)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(vocab_all, output_path)
    print("[SUCCESS] Aggregated Vocab saved to {}!".format(output_path))

    return output_path


if __name__ == "__main__":
    ROOT_DIR = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/re/1000/"
    RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw")

    SPLITS_FNAMES = OrderedDict(
        {
            "train": ["train", "raw_train_data_1000_30days_anony.csv"],
            "val": ["test", "raw_test_data_1000_30days_anony.csv"],
        }
    )
    VOCAB_FNAME = "vocab_1000_vall_30days"
    LABEL = "unplanned_readmission"
    CLASS_IMBALANCE_FNAME = "class_imbalances.json"

    PREPROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "preprocessed")
    TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "training", "data")
    S3_PREPROCESSED_OUTPUT_DIR = (
        "s3://cmsai-mrk-amzn/FinalData/RE/Models/XGBoost/1000/preprocessed/"
    )
    S3_TRAIN_OUTPUT_DIR = (
        "s3://cmsai-mrk-amzn/FinalData/RE/Models/XGBoost/1000/training/data/"
    )

    FOLDS = ["fold_" + str(i) for i in range(5)]
    DATA_ALL = "all"
    NUM_FREQUENT_FEATURES = 300
    NUM_FEATURES_LIST = [100, 200]
    MEDICAL_CODES_ONLY = True

    EXCLUSION_LIST = ["nan", "pad", "unk"] + [LABEL]

    # [START] Integrate all cross-val data and process it.
    print("Aggregating, processing and preparing all data from folds...")
    vocab_path = combine_all_vocabs(RAW_DATA_DIR, FOLDS, DATA_ALL, VOCAB_FNAME)
    vocab = torch.load(vocab_path)

    features = get_frequent_features(
        vocab, NUM_FREQUENT_FEATURES, MEDICAL_CODES_ONLY, EXCLUSION_LIST
    )
    df = get_all_data_from_folds(RAW_DATA_DIR, FOLDS, DATA_ALL)
    df = preprocess(
        df,
        features,
        LABEL,
        DATA_ALL,
        "train",
        PREPROCESSED_DATA_DIR,
        CLASS_IMBALANCE_FNAME,
    )

    prepare(df, NUM_FEATURES_LIST, LABEL, TRAIN_DATA_DIR, DATA_ALL, "train")

    for fold in FOLDS:
        print("Processing and preparing data for {}...".format(fold))
        vocab_path = os.path.join(RAW_DATA_DIR, fold, "vocab", VOCAB_FNAME)
        vocab = torch.load(vocab_path)

        features = get_frequent_features(
            vocab, NUM_FREQUENT_FEATURES, MEDICAL_CODES_ONLY, EXCLUSION_LIST
        )

        for split, fnames in SPLITS_FNAMES.items():
            data_path = os.path.join(RAW_DATA_DIR, fold, fnames[0], fnames[1])
            df = pd.read_csv(data_path)

            df = preprocess(
                df,
                features,
                LABEL,
                fold,
                split,
                PREPROCESSED_DATA_DIR,
                CLASS_IMBALANCE_FNAME,
            )
            prepare(df, NUM_FEATURES_LIST, LABEL, TRAIN_DATA_DIR, fold, split)
            del df

    print("Copying preprocessed data to {}...".format(S3_PREPROCESSED_OUTPUT_DIR))
    command = "aws s3 cp --recursive --quiet {} {}".format(
        PREPROCESSED_DATA_DIR, S3_PREPROCESSED_OUTPUT_DIR
    )
    os.system(command)
    print(
        "[SUCCESS] All preprocessed data is copied {}!".format(
            S3_PREPROCESSED_OUTPUT_DIR
        )
    )

    print("Copying training data to {}...".format(S3_TRAIN_OUTPUT_DIR))
    command = "aws s3 cp --recursive --quiet {} {}".format(
        TRAIN_DATA_DIR, S3_TRAIN_OUTPUT_DIR
    )
    os.system(command)
    print(
        "[SUCCESS] All data ready for model training is copied {}!".format(
            S3_TRAIN_OUTPUT_DIR
        )
    )

"""
A module that preprocesses adverse events and readmissions data to be ready for xgboost training.
"""

import json
import os
import time
import numpy as np
import pandas as pd
#import modin.pandas as pd
import torch
from collections import OrderedDict


def get_frequent_features(vocab, num_features, codes_only=True, exclusion_list=[]): 
    """Get the most frequent codes/features."""
    num_exc = len(exclusion_list) + 100
    features = vocab.freqs.most_common(num_features + num_exc)
    if codes_only:
        features = [word[0] for word in features if word[0] not in exclusion_list and ('_' in word[0])]
    else:
        features = [word[0] for word in features if word[0] not in exclusion_list]
    features = [word for word in features if 'day' not in word] #Exclude day features
    features = features[:num_features]
    return features


def get_feature_ids(vocab, frequent_features):
    """Get the corresponding dictionary ids of the for the selected frequent features."""
    ft_ids = [vocab.stoi[ft] for ft in frequent_features]
    return ft_ids


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


def read_numpy(numpy_path, columns=None):
    """Read numpy file and return dataframe"""
    df = np.load(numpy_path)
    if columns is None:
        df = pd.DataFrame(df, columns=range(1,df.shape[1]+1))
    else:
        df = pd.DataFrame(df, columns=columns)
    return df


def read_labels(labels_path):
    """Read list of labels from path."""
    with open(labels_path, 'r') as fp:
        labels = fp.readlines()
        labels = [label.strip() for label in labels]
    return labels


def get_class_imbalance(df_y):
    """Get class imbalance for all the target variables."""
    imbalance = df_y.apply(lambda x: x.value_counts()).transpose().values.tolist()
    imbalance = dict(zip(df_y.columns.tolist(), imbalance))
    return imbalance


def preprocess(numpy_x_path, numpy_y_path, features, features_ids, labels, split, output_dir, class_imbalance_path=None):
    """Transform the predictor data to one-hot encoding and aggregate with target data."""
    print('Preprocessing and saving {} data...'.format(split))
    df_x = read_numpy(numpy_x_path, columns=None)
    df_y = read_numpy(numpy_y_path, columns=labels)
    
    df_x = df_x.apply(get_one_hot_frequent_features, axis=1, args=(features_ids,))
    df_x = pd.DataFrame(df_x.tolist(), columns=features)
    df = pd.concat([df_x, df_y], axis=1)
    
    if split=='train':
        imb = get_class_imbalance(df_y)
        with open(class_imbalance_path, 'w') as fp:
            json.dump(imb, fp)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, split+'.csv')
    df.to_csv(output_path, index=False)
    print('{} data successfully preprocessed!'.format(split))
    return df


def prepare(df, num_features_list, labels, output_dir, split='train'):
    """Prepares data for model training."""
    num_targets = len(labels)
    features = df.columns.tolist()[:-num_targets]
    for num_features in num_features_list:
        print('Preparing data with {} features...'.format(num_features))
        for label in labels:
            columns = [label] + features[:num_features]
            my_output_dir = os.path.join(output_dir, str(num_features), label)
            
            if not os.path.exists(my_output_dir):
                os.makedirs(my_output_dir)
                
            output_path = os.path.join(my_output_dir, split+'.csv')
            df[columns].to_csv(output_path, index=False, header=None)
    print('Successfully prepared data for training!')


if __name__ == "__main__":
    ROOT_DIR = '/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/ae/1000/'
    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'raw')
    
    SPLITS_FNAMES = OrderedDict({'train': ['final_allvocab_x_train.npy', 'final_allvocab_y_train.npy'],
                                 'val': ['final_allvocab_x_val.npy', 'final_allvocab_y_val.npy'],
                                 'test': ['cms_test_x.npy', 'cms_test_y.npy']
                                })
    
    VOCAB_PATH = os.path.join(RAW_DATA_DIR, 'ae_all_vocab_last180_whole')
    LABELS_PATH = os.path.join(RAW_DATA_DIR, 'labels.txt')
    CLASS_IMBALANCE_PATH = os.path.join(RAW_DATA_DIR, 'class_imbalances.json')

    PREPROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'preprocessed')
    TRAIN_DATA_DIR = os.path.join(ROOT_DIR, 'training')
    S3_OUTPUT_DIR = 's3://cmsai-mrk-amzn/CSVModelInputs/Tes/models/ae/final-global/data/'

    NUM_FREQUENT_FEATURES = 300
    NUM_FEATURES_LIST = [100, 200, 300]
    MEDICAL_CODES_ONLY = True

    labels = read_labels(LABELS_PATH)
    EXCLUSION_LIST = ['nan', 'pad', 'unk'] + labels
    
    vocab = torch.load(VOCAB_PATH)

    features = get_frequent_features(vocab, 
                                     NUM_FREQUENT_FEATURES, 
                                     MEDICAL_CODES_ONLY, 
                                     EXCLUSION_LIST)  
    features_ids = get_feature_ids(vocab, features)
    
    for split, fnames in SPLITS_FNAMES.items():
        data_path_x = os.path.join(RAW_DATA_DIR, fnames[0])
        data_path_y = os.path.join(RAW_DATA_DIR, fnames[1])
        
        df = preprocess(data_path_x, data_path_y, features, features_ids, labels, split, PREPROCESSED_DATA_DIR, CLASS_IMBALANCE_PATH)
        prepare(df, NUM_FEATURES_LIST, labels, TRAIN_DATA_DIR, split)
        del df

    command = 'aws s3 cp --recursive --quiet {} {}'.format(TRAIN_DATA_DIR, S3_OUTPUT_DIR)
    os.system(command)
    print('All data successfully preprocessed and copied to {}!'.format(S3_OUTPUT_DIR))
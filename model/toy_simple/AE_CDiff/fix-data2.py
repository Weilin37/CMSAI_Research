#!/usr/bin/env python
# coding: utf-8

# This script adds `_RF` suffix for each available risk factor in the dataset

# In[50]:


import pandas as pd
import re
import torch


# In[51]:


FNAME = '20110101' #Enter either month/all
DATA_TYPE = 'original' #Enter either downsampled/original

SEQ_LEN = 1000

ALL_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/{FNAME}.csv"
TRAIN_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/train.csv"
VALID_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/val.csv"
TEST_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/test.csv"

OUT_ALL_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/{FNAME}_2.csv"
OUT_TRAIN_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/train2.csv"
OUT_VALID_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/val2.csv"
OUT_TEST_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/test2.csv"

#Ground Truth Risk Factors File Path
GT_CODES_PATH = "../../../data/AE_CDiff_d00845/cdiff_risk_factors_codes.csv"


# In[52]:


df_codes = pd.read_csv(GT_CODES_PATH)
df_codes.head()



def add_rf_suffix(row0, gt_codes_no_star, gt_codes_star):
    """Add _rf suffix to ground truth codes in the given row of the dataset."""
    row = row0.tolist()
    pad = [token for token in row if token == '<pad>']
    row = [token for token in row if token != '<pad>']
    row = [token+'_rf' if token in gt_codes_no_star or list(filter(token.startswith, gt_codes_star)) != [] else token for token in row]
    num_gt_codes = len([token for token in row if token.endswith('_rf')])
    has_gt_codes = 0
    if num_gt_codes > 0:
        has_gt_codes = 1
    row = pad + row + [num_gt_codes, has_gt_codes]
    return row


gt_codes = df_codes.Internal_Code.tolist()
gt_no_stars = [code for code in gt_codes if not code.endswith('*')]
gt_with_stars = [code.replace('*', '') for code in gt_codes if code.endswith('*')]


columns = [str(i) for i in range(SEQ_LEN-1, -1, -1)]
columns2 = columns + ['num_gt_codes', 'has_gt_codes']
nrows = None

#All Data
print('Processing all data...')
df = pd.read_csv(ALL_DATA_PATH, nrows=nrows)
results = df[columns].apply(add_rf_suffix, args=(gt_no_stars, gt_with_stars), axis=1)
if 'has_gt_codes' not in df.columns:
    df['has_gt_codes'] = 0
df[columns2] = results.tolist()
df.to_csv(OUT_ALL_DATA_PATH, index=False)


#Train Data
print('Processing train data...')
df = pd.read_csv(TRAIN_DATA_PATH, nrows=nrows)
results = df[columns].apply(add_rf_suffix, args=(gt_no_stars, gt_with_stars), axis=1)
df[columns2] = results.tolist()
df.to_csv(OUT_TRAIN_DATA_PATH, index=False)


#Valid Data
print('Processing val data...')
df = pd.read_csv(VALID_DATA_PATH, nrows=nrows)
results = df[columns].apply(add_rf_suffix, args=(gt_no_stars, gt_with_stars), axis=1)
df[columns2] = results.tolist()
df.to_csv(OUT_VALID_DATA_PATH, index=False)


#Test Data
print('Processing test data...')
df = pd.read_csv(TEST_DATA_PATH, nrows=nrows)
results = df[columns].apply(add_rf_suffix, args=(gt_no_stars, gt_with_stars), axis=1)
df[columns2] = results.tolist()
df.to_csv(OUT_TEST_DATA_PATH, index=False)
print('SUCCESS!')

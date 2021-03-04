#!/usr/bin/env python
# coding: utf-8

# This script replaces `d_s` or `d_S` with `d_` only

# In[1]:

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1457MiB |
# |    1   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    2   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    3   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    4   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    5   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    6   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# |    7   N/A  N/A     94481      C   ...vs/pytorch_p36/bin/python     1423MiB |
# +-----------------------------------------------------------------------------+


import pandas as pd
import re


# In[2]:


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


# In[3]:


def fix_token(row0):
    """Replace d_s and d_S with d_ only for the given data row."""
    row = row0.tolist()
    row = [re.sub('d_s', 'd_', token, flags=re.IGNORECASE) for token in row]
    return row


# In[ ]:


columns = [str(i) for i in range(SEQ_LEN-1, -1, -1)]
nrows = None

#All Data
df = pd.read_csv(ALL_DATA_PATH, nrows=nrows)
results = df[columns].apply(fix_token, axis=1)
df[columns] = results.tolist()
df.to_csv(OUT_ALL_DATA_PATH, index=False)
print(df.shape)
df.head()


# In[ ]:


#Train Data
df = pd.read_csv(TRAIN_DATA_PATH, nrows=nrows)
results = df[columns].apply(fix_token, axis=1)
df[columns] = results.tolist()
df.to_csv(OUT_TRAIN_DATA_PATH, index=False)
print(df.shape)
df.head()


# In[ ]:


#Valid Data
df = pd.read_csv(VALID_DATA_PATH, nrows=nrows)
results = df[columns].apply(fix_token, axis=1)
df[columns] = results.tolist()
df.to_csv(OUT_VALID_DATA_PATH, index=False)
print(df.shape)
df.head()


# In[ ]:


#Test Data
df = pd.read_csv(TEST_DATA_PATH, nrows=nrows)
results = df[columns].apply(fix_token, axis=1)
df[columns] = results.tolist()
df.to_csv(OUT_TEST_DATA_PATH, index=False)
print(df.shape)
df.head()


# In[ ]:





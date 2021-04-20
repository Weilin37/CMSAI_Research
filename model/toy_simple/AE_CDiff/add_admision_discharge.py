#!/usr/bin/env python
# coding: utf-8

# The script inserts admission/discharge events to the corresponding positions and saves as output


#import ray
#ray.init(num_cpus=8)
#import modin.pandas as pd
import pandas as pd

#import pandas as pd
import re
import os
import json


# In[3]:


FNAME = '20110101' #Enter either month/all
DATA_TYPE = 'original' #Enter either downsampled/original

SEQ_LEN = 1000

AD_DIS_PATH = "../../../data/AE_CDiff_d00845/output/data/1000/admission_discharge/results.json"

ALL_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/{FNAME}.csv"
TRAIN_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/train.csv"
VALID_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/val.csv"
TEST_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/test.csv"

OUT_ALL_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed_v2/{FNAME}.csv"
OUT_TRAIN_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed_v2/splits/{FNAME}/train.csv"
OUT_VALID_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed_v2/splits/{FNAME}/val.csv"
OUT_TEST_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed_v2/splits/{FNAME}/test.csv"

label = 'd_00845'

with open(AD_DIS_PATH, "r") as fp:
    dict_ad_dis = json.load(fp)

def add_ad_dis(row0, dict_ad_dis, n_events=1000, date=''):
    """Add admission and discharge to the row if available."""
    patient_id = row0['patient_id']
    if date:
        patient_id = patient_id + "_" + date
    ad_dis = dict_ad_dis[patient_id]
    ad = ad_dis['admission']
    dis = ad_dis['discharge']
    cols = [str(i) for i in range(n_events-1, -1, -1)]
    all_cols = list(row0.index)
    if ad or dis:
        row = row0[cols].tolist()
        row.reverse()
        for indx in ad:
            row.insert(indx, 'admission')
        for indx in dis:
            row.insert(indx, 'discharge')
        row = row[:n_events]
        row.reverse()
        row0[cols] = row[:]
    return row0    

output_dir2 = os.path.dirname(OUT_TRAIN_DATA_PATH)
os.makedirs(output_dir2, exist_ok=True)

print('Adding admissions/discharge to original data...')
print('Processing train data...')
df_train = pd.read_csv(TRAIN_DATA_PATH, nrows=None)
df_train2 = df_train.apply(add_ad_dis, args=(dict_ad_dis, SEQ_LEN, FNAME), axis=1)
df_train2.to_csv(OUT_TRAIN_DATA_PATH, index=False)
del df_train, df_train2

print('Processing val data...')
df_val = pd.read_csv(VALID_DATA_PATH, nrows=None)
df_val2 = df_val.apply(add_ad_dis, args=(dict_ad_dis, SEQ_LEN, FNAME), axis=1)
df_val2.to_csv(OUT_VALID_DATA_PATH, index=False)
del df_val, df_val2

print('Processing test data...')
df_test = pd.read_csv(TEST_DATA_PATH, nrows=None)
df_test2 = df_test.apply(add_ad_dis, args=(dict_ad_dis, SEQ_LEN, FNAME), axis=1)
df_test2.to_csv(OUT_TEST_DATA_PATH, index=False)

print('Success!')
#df_train[label].value_counts(normalize=False), df_train[label].value_counts(normalize=True)
#df_val[label].value_counts(normalize=False), df_val[label].value_counts(normalize=True)
#df_test[label].value_counts(normalize=False), df_test[label].value_counts(normalize=True)


#df_all2.to_csv(OUT_ALL_DATA_PATH, index=False)


# In[ ]:





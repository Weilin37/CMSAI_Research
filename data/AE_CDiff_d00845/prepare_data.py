#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
# **Author: Tesfagabir Meharizghi *(Adopted from Lin Lee Notebook)*<br>
# Last Updated: 02/12/2021**
# 
# Notebook for prepares and splits the AE data with only one target variable:
# - Target: d_00845 for C.Diff Infection (ICD-9 code C.Diff is  008.45)
# - C.Diff is selected because main events/causes are relatively well known so that the features importances predicted from different algorithms and models could be compared with the ground truth
# - Actions:
#     - Read 365 data
#     - Flattens into 1000 events
#     - Downsamples the data to make it balanced
#     - Combine all months data
#     - Splits data into train/val/test
#     - Finally saves them to disk

# ## 0. Install packages - First time only

# In[1]:


# pip install nb-black


# In[2]:



# In[3]:


##Import packages
import numpy as np
import pandas as pd
import os
from more_itertools import unique_everseen

from sklearn.model_selection import train_test_split


# In[4]:


def flatten(x, n_events=1000):
    """Flatten the 365 dataset into N long events"""

    def get_days(x):
        """Calculate number of days between events"""
        new_lst = []
        counter = 1
        counting = False
        for event in x:
            if event is np.nan or (type(event) == float and math.isnan(event)):
                if not counting:
                    counting = True
                counter += 1
            else:

                if counting:
                    counting = False
                    try:
                        event = f"{counter + 1}_days," + event
                    except:
                        print(type(counter), counter)
                        print(event, type(event))
                    new_lst.append(event)
                    counter = 0
                else:
                    event = "1_days," + event
                    new_lst.append(event)

        return new_lst

    # count days with no events, move admission/discharge to the end of the day, dedupe events per day
    x = np.array(get_days(x))
    lst = [move_ad_dis(str(day).replace(" ", "").split(",")) for day in x.ravel("K")]

    # flatten, clean up corner cases
    lst = [event for day in lst for event in day]
    if not lst:
        return ["<pad>"] * (n_events - len(lst)) + lst

    if "_days" in lst[0]:
        lst = lst[1:]

    if len(lst) >= n_events:
        return lst[-n_events:]

    return ["<pad>"] * (n_events - len(lst)) + lst


def move_ad_dis(events_in_day):
    """Move target_event and patient_id to the end of the list, dedupe events"""
    if not isinstance(events_in_day, list):
        return events_in_day

    events_in_day = list(unique_everseen(events_in_day))
    has_admission = False
    has_discharge = False

    if "admission" in events_in_day:
        has_admission = True
        events_in_day.remove("admission")

    if "discharge" in events_in_day:
        has_discharge = True
        events_in_day.remove("discharge")

        if has_admission:
            events_in_day.append("admission")

        if has_discharge:
            events_in_day.append("discharge")

    return events_in_day


def get_flat_df(raw_df, x_lst, copy_lst, n_events):
    """
    Function to flatten dataframe into 1000 long sequence.

    Calls function flatten, which in turn calls move_ad_dis
    """
    columns = [str(x) for x in range(n_events - 1, -1, -1)]
    flat_df = pd.DataFrame(
        raw_df[x_lst]
        .apply(
            flatten,
            args=(n_events,),
            axis=1,
        )
        .tolist(),
        columns=columns,
    )

    for colname in copy_lst:
        flat_df[colname] = raw_df[colname].tolist()

    return flat_df


# Downsample Data to make it balanced
# Class count
def downsample(df0, label):
    """Downsample dataset to make classes balanced."""
    df = df0.copy()
    count_class_0, count_class_1 = df[label].value_counts()

    # Divide by class
    df_class_0 = df[df[label] == 0]
    df_class_1 = df[df[label] == 1]

    df_class_0_under = df_class_0.sample(count_class_1)

    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    df = df.sample(frac=1)  # shuffle
    return df


def split_data(df, test_size, label, output_dir, n_events=1000):
    """Split data into train/val/test sets. test_size is the fraction of val/test sets"""
    feature_names = [str(x) for x in range(n_events - 1, -1, -1)] + ['patient_id']
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        df[feature_names],
        df[label],
        test_size=2 * test_size,
        stratify=df[label],
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test, test_size=0.5
    )
    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_val, y_val], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return df_train, df_val, df_test


# ## Preprocess Data

# In[ ]:


NROWS = 10#None 

N_DAYS = 365  # Number of input days
N_EVENTS = 1000  # Output number of events

X_INPUT_LST = [
    str(x) for x in range(N_DAYS, -1, -1)
]  # total days in datasets, usually 365.
LABEL = "d_00845"
UID_COLUMN = "patient_id"
COPY_LIST = [LABEL, UID_COLUMN]

SPLIT_TEST_SIZE = 0.15  # 70/15/15 splits

RAW_DATA_DIR = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/anonymize/AE/Data/Anonymized/365NoDeath/"
OUTPUT_ORIGINAL_DIR = "./output/data/1000/original_v2/"

OUTPUT_DOWNSAMPLED_DIR = "./output/data/1000/downsampled_v2/"

os.makedirs(OUTPUT_ORIGINAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DOWNSAMPLED_DIR, exist_ok=True)

df_all = None
df_down_all = None
for i in range(1, 11):
    MONTH = f"2011{i:02}01"
    print(f"Processing data for Month: {MONTH}....")

    IN_FNAME = f"ae_patients_365_{MONTH}.csv"
    OUT_FNAME = f"{MONTH}.csv"

    raw_data_path = os.path.join(RAW_DATA_DIR, IN_FNAME)
    flat_data_path = os.path.join(OUTPUT_ORIGINAL_DIR, OUT_FNAME)
    flat_downsampled_path = os.path.join(OUTPUT_DOWNSAMPLED_DIR, OUT_FNAME)

    df_raw = pd.read_csv(raw_data_path, low_memory=False, nrows=NROWS)

    df_flat = get_flat_df(df_raw, X_INPUT_LST, COPY_LIST, N_EVENTS)

    #df_flat = pd.read_csv(flat_data_path)
    print(f"Flat Shape = {df_flat.shape}")
    df_down = pd.read_csv(flat_downsampled_path)
    print(f"Downsampled Shape = {df_down.shape}")
    
    #df_down = pd.read_csv(flat_downsampled_path)
    #print(f"Downsampled Shape = {df_down.shape}")
    df_down = downsample(df_flat, LABEL)
    print(f"Flat Shape = {df_flat.shape}, Downsampled Shape = {df_down.shape}")

    # Save the data
    df_flat.to_csv(flat_data_path, index=False)
    df_down.to_csv(flat_downsampled_path, index=False)

    # Split data
    output_dir = os.path.join(OUTPUT_ORIGINAL_DIR, f"splits/{MONTH}/")
    os.makedirs(output_dir, exist_ok=True)
    _ = split_data(df_flat, SPLIT_TEST_SIZE, LABEL, output_dir, n_events=N_EVENTS)

    output_dir = os.path.join(OUTPUT_DOWNSAMPLED_DIR, f"splits/{MONTH}/")
    os.makedirs(output_dir, exist_ok=True)
    _ = split_data(df_down, SPLIT_TEST_SIZE, LABEL, output_dir, n_events=N_EVENTS)

    # Combine data
    #df_flat[UID_COLUMN] = df_flat[UID_COLUMN] + f"_{MONTH}"
    #df_down[UID_COLUMN] = df_down[UID_COLUMN] + f"_{MONTH}"
    #if df_all is None:
    #if df_down_all is None:
        #df_all = df_flat.copy()
        #df_down_all = df_down.copy()
    #else:
        #df_all = pd.concat([df_all, df_flat], axis=0)
        #df_down_all = pd.concat([df_down_all, df_down], axis=0)
    #del df_flat
    #del df_raw, df_flat, df_down
    print("*" * 50)

all_fname = "all.csv"
#all_data_path = os.path.join(OUTPUT_ORIGINAL_DIR, all_fname)
all_down_data_path = os.path.join(OUTPUT_DOWNSAMPLED_DIR, all_fname)

# Shuffle dataset
#df_all = df_all.sample(frac=1)
#df_down_all = df_down_all.sample(frac=1)
#print("all_flat_data shape: ", df_all.shape)

df_down_all = pd.read_csv(all_down_data_path)
print("all_down_data shape: ", df_down_all.shape)

# Split data
#output_dir = os.path.join(OUTPUT_ORIGINAL_DIR, "splits/all")
#os.makedirs(output_dir, exist_ok=True)
#_ = split_data(df_all, SPLIT_TEST_SIZE, LABEL, output_dir, n_events=N_EVENTS)

output_dir = os.path.join(OUTPUT_DOWNSAMPLED_DIR, "splits/all")
os.makedirs(output_dir, exist_ok=True)
_ = split_data(df_down_all, SPLIT_TEST_SIZE, LABEL, output_dir, n_events=N_EVENTS)

# Save data
#df_all.to_csv(all_data_path, index=False)
#df_down_all.to_csv(all_down_data_path, index=False)
print("SUCCESS!")

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_gt_code_patient(row, gt_codes, seq_len=1000):
    """Get patient_id of those having the gt_codes"""
    cols = [str(i) for i in range(seq_len - 1, -1, -1)]
    num_common = len(set(row[cols]).intersection(gt_codes))
    return num_common


def split_data_v2(df, test_size, label, output_dir, n_events=1000):
    """Split data into train/val/test sets. test_size is the fraction of val/test sets"""
    df['has_gt_codes'] = (df['num_gt_codes']>0).astype(int)
    stratify_cols = [label, 'has_gt_codes']
    df_train, df_val_test = train_test_split(
            df,
            test_size=2 * test_size,
            stratify=df[stratify_cols])

    df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            stratify=df_val_test[stratify_cols])
    
    os.makedirs(output_dir, exist_ok=True)
    
    #Sort val & test
    df_val = df_val.sort_values('has_gt_codes', ascending=False)
    df_test = df_test.sort_values('has_gt_codes', ascending=False)
    
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    return df_train, df_val, df_test


SEQ_LEN = 1000
TEST_SIZE = 0.15
LABEL = 'd_00845'

ALL_DOWN_DATA_PATH = (
    f"./output/data/1000/downsampled/all.csv"
)

MONTH_DATA_PATH = (
    f"./output/data/1000/original/20110101.csv"
)

GT_CODES_PATH = './cdiff_risk_factors_codes.csv'

ALL_DOWN_DATA_OUT_PATH = f"./output/data/1000/downsampled/preprocessed/all.csv"

MONTH_DATA_OUT_PATH = (
    f"./output/data/1000/original/preprocessed/20110101.csv"
)

#Read GT Codes
gt_codes = pd.read_csv(GT_CODES_PATH)
gt_codes = set(gt_codes.Internal_Code)
print('Total GT Codes:', len(gt_codes))

df = pd.read_csv(ALL_DOWN_DATA_PATH)
print("Original shape:", df.shape)
# Remove empty rows
df = df[df["0"] != "<pad>"]
print("Without Empty shape:", df.shape)

df["num_gt_codes"] = df.apply(get_gt_code_patient, args=(gt_codes, SEQ_LEN), axis=1)
print("Total_patients with GT Codes:", sum(df["num_gt_codes"] > 0))

print(df[df["num_gt_codes"] != 0]["d_00845"].value_counts())
df[df["num_gt_codes"] != 0]["d_00845"].value_counts().plot.bar()

output_dir = os.path.dirname(ALL_DOWN_DATA_OUT_PATH)
os.makedirs(output_dir, exist_ok=True)
df.to_csv(ALL_DOWN_DATA_OUT_PATH, index=False)

print('Spliting data...')
output_dir, fname = ALL_DOWN_DATA_OUT_PATH.rsplit('/', 1)
fname = fname.split('.')[0]
output_dir = os.path.join(output_dir, 'splits', fname)
_ = split_data_v2(df, TEST_SIZE, LABEL, output_dir)

print(f'Preprocessed data successfully saved to {ALL_DOWN_DATA_OUT_PATH}')
print('-'*30)

df = pd.read_csv(MONTH_DATA_PATH)
print("Original shape:", df.shape)
# Remove empty rows
df = df[df["0"] != "<pad>"]
print("Without Empty shape:", df.shape)

df["num_gt_codes"] = df.apply(get_gt_code_patient, args=(gt_codes, SEQ_LEN), axis=1)
print("Total_patients with GT Codes:", sum(df["num_gt_codes"] > 0))

print(df[df["num_gt_codes"] != 0]["d_00845"].value_counts())
df[df["num_gt_codes"] != 0]["d_00845"].value_counts().plot.bar()

output_dir = os.path.dirname(MONTH_DATA_OUT_PATH)
os.makedirs(output_dir, exist_ok=True)
df.to_csv(MONTH_DATA_OUT_PATH, index=False)

print('Spliting data...')
output_dir, fname = MONTH_DATA_OUT_PATH.rsplit('/', 1)
fname = fname.split('.')[0]
output_dir = os.path.join(output_dir, 'splits', fname)
_ = split_data_v2(df, TEST_SIZE, LABEL, output_dir)
print(f'Preprocessed data successfully saved to {MONTH_DATA_OUT_PATH}')
"""Module to select few sequenced-based test examples of positive label and negative label used for SHAP scores visualization."""

import os
import pandas as pd
from collections import Counter


if __name__ == "__main__":
    SEQ_LEN = 300
    
    NUM_POS_PATIENTS = 2
    NUM_NEG_PATIENTS = 2

    DATA_DIR = f"/home/ec2-user/SageMaker/CMSAI_Research/data/toy_dataset/data/seq_final_v2/{SEQ_LEN}/"
    
    INPUT_PATH = os.path.join(DATA_DIR, "test.csv")
    OUTPUT_PATH = os.path.join(DATA_DIR, "visualized_test_patients.txt")

    print(f'Selecting Patients for seq_len={SEQ_LEN}...')
    #Read data
    df = pd.read_csv(INPUT_PATH)
    
    #Selecting patients
    num_rows = df.shape[0]
    pos_patients = []
    neg_patients = []
    for i in range(num_rows):
        num_adverse = 0
        num_helpers = 0
        num_unhelpers = 0
        #feature_names = [str(j) for j in range(SEQ_LEN-1, -1, -1)]
        row = df.iloc[i]
        freqs = Counter(row)
        patient_id = df["patient_id"].iloc[i]
        if (row['seq_event'] == 'UHA') and (row['label']==1):
            if (len(pos_patients) < NUM_POS_PATIENTS):
                pos_patients.append(patient_id)
        elif (row['seq_event'] == 'AHU') and (row['label']==0):
            if (len(neg_patients) < NUM_NEG_PATIENTS):
                neg_patients.append(patient_id)
        if (len(pos_patients) >= NUM_POS_PATIENTS) and (len(neg_patients) >= NUM_NEG_PATIENTS):
            break
    selected_patients = pos_patients + neg_patients
    
    #Writing patients to file
    with open(OUTPUT_PATH, "w") as fp:
        fp.write("\n".join(selected_patients))
    
    print(f'Successfully Selected and Saved to {OUTPUT_PATH}!')

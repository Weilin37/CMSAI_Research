# CMSAI_Research
This repository contains source code related to Merck/CMSAI PoC Engagement.

To get more information about the submitted paper associated with the engagement, please [click here](##Paper/Link/here)

It is organized as shown below.

# Directory Structure
```
.
├── competition ==> CMSAI Competition related scripts
│   ├── data ==> Data preparation scripts
│   │   ├── ae ==> For Adverse Events (AE) data
│   │   └── readmission ==> For hospital readmissions data
│   └── model ==> Model development scripts
│       ├── ae ==> AE-based models
│       └── readmission ==> Readmissions-based models
├── explainability ==> Model Explainability related scripts
│   ├── data ==> Data Preparation Scripts
│   │   ├── ae_cdiff ==> For AE CDiff dataset
│   │   └── synthetic ==> Synthetic dataset generation script
│   └── model ==> Model training and explainability related scripts
│       ├── deep_id_pytorch.py ==> Script to compute deep-learning based shap scores
│       ├── imp_utils.py ==> Utils for computing importance scores
│       ├── lstm_att_models.py ==> Attention-based model definition
│       ├── lstm.ipynb ==> LSTM-based model training and feature importance scores computation
│       ├── lstm_lrp_all_data.py ==> Script to compute LSTM LRP for all test/val data
│       ├── lstm_lrp_models.py ==> LRP compatible LSTM model definition
│       ├── lstm_lrp_shap_all_data.ipynb ==> Merge all LSTM's LRP and SHAP scores
│       ├── lstm_models.py ==> Simple-LSTM model definition
│       ├── lstm_self_att_models.py ==> Self-attention Based LSTM model definition
│       ├── lstm_shap_all_data.py ==> Script to compute LSTM SHAP for all test/val data
│       ├── lstm_utils.py ==> Util functions LSTM-based model development
│       ├── utils.py ==> Utils functions
│       ├── xgb.ipynb ==> XGB model training and feature importance scores computation
│       └── xgb_utils.py ==> Util functions XGB model development
└── README.md
```
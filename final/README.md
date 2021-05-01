# CMSAI_Research
This repository contains source code related to Merck/CMSAI PoC Engagement.

# Code Organization
It is organized in two different but similar directories:
* [competition/](competition/): Contains all the code used during the CMSAI challenge on predicting adverse events and hospital readmissions from claims data. 
    * [competition/data/](competition/data/): contains codes related to data preparation and preprocessing
    * [competition/model/](competition/model/): contain codes related to models development (XGBoost, LSTM, Transformer)
* [explainability/](explainability/): stores the code related not only to models development (XGBoost & Attention-based LSTM) but also to computing feature importances using different explainability methods (SHAP, LRP, Attention, etc.). You can read this [conference paper](link/here/) to get more insights about this work.
    * [explainability//data/](explainability//data/): contains codes related to data preparation and preprocessing
    * [explainability//model/](explainability//model/): contain codes related to models development & computing the feature importances.

The detailed code organization is shown below.

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
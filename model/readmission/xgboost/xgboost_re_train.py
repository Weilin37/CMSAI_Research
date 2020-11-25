"""
A module that trains adverse events xgboost models.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import shutil

from time import gmtime, strftime

import sagemaker
import boto3
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

# import shap
import tarfile
import pickle

pd.options.mode.chained_assignment = None

from sagemaker.image_uris import retrieve

from pprint import pprint

import xgboost as xgb


def load_class_imbalances(class_imbalances_path):
    """Load class imbalances from json file."""
    with open(class_imbalances_path, 'r') as fp:
        class_imbalances = json.load(fp)
    return class_imbalances


def get_best_model_path(tuning_job_result): 
    """Gets model path in the S3 from the tuning job outputs
    Args:
        tuning_job_result(object): Hyperparameter tuning result
    Returns:
        str: Best model path from the tuning jobs
    """ 
    best_job = tuning_job_result.get('BestTrainingJob',None) 
    job_name = best_job['TrainingJobName'] 
    model_name=job_name + '-model' 
    info = smclient.describe_training_job(TrainingJobName=job_name) 
    model_path = info['ModelArtifacts']['S3ModelArtifacts'] 
    return model_path


def get_tuner_status_and_result_until_completion(tuner, num_features, target, sleep_time=60):
    """Print results of running tuner on a regular interval until completion
    Args:
        tuner(Object): The running Hyperparameter tuner object
        target(str): Target string
    Returns:
        None
    """
    print('Training for #features={} and class={}...'.format(num_features, target))
    while True:
        tuning_job_result = smclient.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuner.latest_tuning_job.job_name)
        job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
        status = tuning_job_result['HyperParameterTuningJobStatus']

        auc_value = None
        if tuning_job_result.get('BestTrainingJob',None):
            best_job = tuning_job_result['BestTrainingJob']
            metric = best_job['FinalHyperParameterTuningJobObjectiveMetric']['MetricName']
            auc_value = best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']
            auc_value = round(auc_value, 4)
            print("Total jobs completed: {}".format(job_count))
            print("Metric: {}".format(metric))
            print("Best AUC: {}".format(auc_value))
        else:
            print('-')
            
        if status == 'Completed':
            model_path = get_best_model_path(tuning_job_result)
            print('Training for #features={} class={} completed!'.format(num_features, target))
            print('==================================')
            return auc_value, model_path
        
        time.sleep(sleep_time)


def train_hpo(hyperparameter_ranges, container, execution_role, instance_count, instance_type, 
              output_path, sagemaker_session, eval_metric, objective, objective_metric_name, 
              max_train_jobs, max_parallel_jobs, scale_pos_weight, train_channel, val_channel=None):
    """Train a model based on a given data fold and HPO training job summary job."""
    import pdb; pdb.set_trace()
    xgb_model = sagemaker.estimator.Estimator(container,
                                        execution_role, 
                                        instance_count=instance_count, 
                                        instance_type=instance_type,
                                        output_path=output_path,
                                        sagemaker_session=sagemaker_session)

    xgb_model.set_hyperparameters(eval_metric=eval_metric,
                            objective=objective,
                            scale_pos_weight=scale_pos_weight, #For class imbalance
                            num_round=200,
                            rate_drop=0.3,
                            max_depth=5,
                            subsample=0.8,
                            gamma=2,
                            eta=0.2)

    tuner = HyperparameterTuner(xgb_model, 
                                objective_metric_name,
                                hyperparameter_ranges,
                                max_jobs=max_train_jobs,
                                max_parallel_jobs=max_parallel_jobs)
        
    data_channels = {'train': train_channel, 'validation': val_channel}
    tuner.fit(inputs=data_channels)
    
    return tuner


def train_model(hpo_summary_row, container, execution_role, instance_count, instance_type, output_path, 
                sagemaker_session, eval_metric, objective, scale_pos_weight, train_channel, val_channel=None, job_name=None):
    """Train a model based on a given data fold and HPO training job summary job."""
    params = dict(hpo_summary_row.iloc[:12])
    
    if job_name is None:
        xgb_model = sagemaker.estimator.Estimator(container,
                                            execution_role, 
                                            instance_count=instance_count, 
                                            instance_type=instance_type,
                                            output_path=output_path,
                                            sagemaker_session=sagemaker_session)
    else:
        xgb_model = sagemaker.estimator.Estimator(container,
                                            execution_role, 
                                            instance_count=instance_count, 
                                            instance_type=instance_type,
                                            output_path=output_path,
                                            sagemaker_session=sagemaker_session,
                                            base_job_name=job_name)
    
    xgb_model.set_hyperparameters(eval_metric=eval_metric,
                            objective=objective,
                            scale_pos_weight=scale_pos_weight, #For class imbalance
                            **params)
    
    data_channels = {'train': train_channel, 'validation': val_channel}
    xgb_model.fit(inputs=data_channels)
    
    return xgb_model


def get_best_model_params(hpo_results_path):
    """Get the parameters for the best training job."""
    df_train_results = pd.read_csv(hpo_results_path)
    avg_performances = df_train_results.Avg.values.tolist()
    max_indx = avg_performances.index(max(avg_performances))
    params = df_train_results.iloc[max_indx, :12]
    params = dict(params)
    return params


if __name__ == "__main__":
    #Number of features used for training
    NUM_FEATURES = 100
    FOLDS = ['fold_'+ str(i) for i in range(4)]
    TUNING_FOLD = 0 #Fold number to be used to launch the HPO tuning job
    DATA_ALL = 'all'
    LABEL = 'unplanned_readmissions'
    ROOT_DIR = '/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/re/1000/'
    DATA_DIR = os.path.join(ROOT_DIR, 'preprocessed')
    TRAIN_DIR = os.path.join(DATA_DIR, 'training')
    CLASS_IMBALANCE_PATH_PATTERN = os.path.join(DATA_DIR, '{}', 'class_imbalances.json')
    HPO_SUMMARY_PATH_PATTERN = os.path.join(TRAIN_DIR, str(NUM_FEATURES), '{}')
    HPO_RESULTS_PATH = os.path.join(TRAIN_DIR, str(NUM_FEATURES), DATA_ALL, 'hpo_results.csv')
    TRAIN_RESULTS_PATH = os.path.join(TRAIN_DIR, str(NUM_FEATURES), DATA_ALL, 'hpo_results.csv')    

    #Bucket where the trained model is stored
    BUCKET = 'cmsai-mrk-amzn'
    #Directory prefix where the model training outputs is saved
    now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    DATA_PREFIX = 'FinalData/RE/Models/XGBoost/1000/training/data' #data_split, num_features
    MODEL_PREFIX = 'FinalData/RE/Models/XGBoost/1000/training/models' #data_split, num_features, time
    
    ###Algorithm config
    ALGORITHM = 'xgboost'
    REPO_VERSION = '1.2-1'

    ###HPO/training job config
    TRAIN_INSTANCE_TYPE = 'ml.m4.16xlarge'
    TRAIN_INSTANCE_COUNT = 2
    MAX_PARALLEL_JOBS = 2#4
    MAX_TRAIN_JOBS = 2#4#20 #TODO: Update later
    
    EVALUATION_METRIC = 'auc'
    OBJECTIVE = 'binary:logistic'
    OBJECTIVE_METRIC_NAME = 'validation:auc'

    #Update hyperparameter ranges
#     HYPERPARAMETER_RANGES = {'eta': ContinuousParameter(0, 1),
#                             'alpha': ContinuousParameter(0, 2),
#                             'max_depth': IntegerParameter(1, 10)}
    
    HYPERPARAMETER_RANGES = {'eta': ContinuousParameter(0.1, 0.5),
                           'alpha': ContinuousParameter(0, 2),
                           'max_depth': IntegerParameter(1, 10),
                           'gamma': ContinuousParameter(0, 5),
                           'num_round': IntegerParameter(200, 500),
                           'colsample_bylevel': ContinuousParameter(0.1, 1.0),
                           'colsample_bynode': ContinuousParameter(0.1, 1.0),
                           'colsample_bytree': ContinuousParameter(0.5, 1.0),
                           'lambda': ContinuousParameter(0, 1000),
                           'max_delta_step': IntegerParameter(0, 10),
                           'min_child_weight': ContinuousParameter(0, 120),
                           'subsample': ContinuousParameter(0.5, 1.0),
                           }


    ### SageMaker Initialization
    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()
    smclient = boto3.Session().client('sagemaker')

    sess = sagemaker.Session()

    container = retrieve(ALGORITHM, region, version=REPO_VERSION)

    training_results = []
    s3_resource = boto3.Session().resource('s3')
    #Model Selection...
    print('Launching model selection using cross validation...')
    for i, fold in enumerate(FOLDS):
        
        #Prepare the input train & validation data path
        s3_train_path = 's3://{}/{}/{}/{}/train'.format(BUCKET, DATA_PREFIX, NUM_FEATURES, fold)
        s3_val_path = 's3://{}/{}/{}/{}/val'.format(BUCKET, DATA_PREFIX, NUM_FEATURES, fold)
        s3_input_train = sagemaker.inputs.TrainingInput(s3_data=s3_train_path, content_type='csv')
        s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=s3_val_path, content_type='csv')
        s3_output_path = 's3://{}/{}/{}/{}/{}/output'.format(BUCKET, MODEL_PREFIX, now, NUM_FEATURES, fold)

        #Load class imbalances
        class_imbalance_path = CLASS_IMBALANCE_PATH_PATTERN.format(fold)
        class_imbalances = load_class_imbalances(class_imbalance_path)
        imb = class_imbalances[LABEL]
        scale_pos_weight = float(imb[0])/imb[1] # negative/positive

        if i == TUNING_FOLD:
            print('Launching HPO tuning job with {} data...'.format(fold))
            tuner = train_hpo(hyperparameter_ranges=HYPERPARAMETER_RANGES, 
                              container=container, 
                              execution_role=role, 
                              instance_count=TRAIN_INSTANCE_COUNT, 
                              instance_type=TRAIN_INSTANCE_TYPE, 
                              output_path=s3_output_path, 
                              sagemaker_session=sess, 
                              eval_metric=EVALUATION_METRIC, 
                              objective=OBJECTIVE, 
                              objective_metric_name=OBJECTIVE_METRIC_NAME, 
                              max_train_jobs=MAX_TRAIN_JOBS, 
                              max_parallel_jobs=MAX_PARALLEL_JOBS, 
                              scale_pos_weight=scale_pos_weight, 
                              train_channel=s3_input_train, 
                              val_channel=s3_input_validation)
                
            #Save the HPO tuning job summary data
            job_name = tuner.latest_tuning_job.name
            my_tuner = sagemaker.HyperparameterTuningJobAnalytics(job_name)
            df = my_tuner.dataframe()
            hpo_summary_path = HPO_SUMMARY_PATH_PATTERN.format(fold)
            
            hpo_summary_dir = os.path.dirname(hpo_summary_path)
            if not os.path.exists(hpo_summary_dir):
                os.makedirs(hpo_summary_dir)    
            df.to_csv(hpo_summary_path, index=False)
            
            training_results.append(df['FinalObjectiveValue'].values.tolist())
            
        else:
            print('Launching training job with {} data...'.format(fold))
            hpo_summary_path = HPO_SUMMARY_PATH_PATTERN.format(FOLDS[TUNNING_FOLD])
            df_summary = pd.read_csv(hpo_summary_path)
            my_args = (container, role, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, s3_output_path, 
                       sess, EVALUATION_METRIC, OBJECTIVE, scale_pos_weight, s3_input_train, s3_input_validation)
            result = df_summary.apply(train_model, axis=1, args=my_args)
            training_results.append(result)
            
    hpo_summary_path = HPO_SUMMARY_PATH_PATTERN.format(FOLDS[TUNNING_FOLD])
    df_tuning_summary = pd.read_csv(hpo_summary_path)
    df_params = df_tuning_summary.iloc[:, :12] #Hyperparameters

    df_results = pd.DataFrame(training_results, columns=FOLDS)
    df_results['Avg'] = df_results.mean(axis=0)
    
    df_final_results = pd.concat([df_params, df_results], axis=1)
    hpo_results_dir = os.path.dirname(HPO_RESULTS_PATH)
    if not os.path.exists(hpo_results_dir):
        os.makdirs(hpo_results_dir)
    df_final_results.to_csv(HPO_RESULTS_PATH)    
    print('Model Selection Successfully Done!')
    
    
    #Best Model Training...
    print('Launching the best model training using all data...')
    #Get parameters for the best training job
    params = get_best_model_params(HPO_RESULTS_PATH)
    
    #Prepare the input train & validation data path
    s3_train_path = 's3://{}/{}/{}/{}/train'.format(BUCKET, DATA_PREFIX, NUM_FEATURES, DATA_ALL)
    s3_val_path = 's3://{}/{}/{}/{}/val'.format(BUCKET, DATA_PREFIX, NUM_FEATURES, DATA_ALL)
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=s3_train_path, content_type='csv')
    s3_input_validation = sagemaker.inputs.TrainingInput(s3_data=s3_val_path, content_type='csv')
    s3_output_path = 's3://{}/{}/{}/{}/{}/output'.format(BUCKET, MODEL_PREFIX, now, NUM_FEATURES, DATA_ALL)

    #Load class imbalances
    class_imbalance_path = CLASS_IMBALANCE_PATH_PATTERN.format(DATA_ALL)
    class_imbalances = load_class_imbalances(class_imbalance_path)
    imb = class_imbalances[LABEL]
    scale_pos_weight = float(imb[0])/imb[1] # negative/positive

    my_args = (container, role, TRAIN_INSTANCE_COUNT, TRAIN_INSTANCE_TYPE, s3_output_path, 
               sess, EVALUATION_METRIC, OBJECTIVE, scale_pos_weight, s3_input_train, s3_input_validation)
    model_path = os.path.join(output_path, 'model.tar.gz')
    val_auc = train_model(*my_args)
    
    columns = ['class', 'num_features', 'val_auc', 'best_model_path']
    eval_results = [label, NUM_FEATURES, val_auc, model_path]
    df_eval_results = pd.DataFrame(eval_results, columns=columns)

    df_eval_results.to_csv(TRAIN_RESULTS_PATH, index=False)
    print('Best Training Model Successfully Trained!')
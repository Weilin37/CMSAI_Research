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
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)

# import shap
import tarfile
import pickle

pd.options.mode.chained_assignment = None

from sagemaker.image_uris import retrieve

from pprint import pprint

import xgboost as xgb


def load_labels(labels_path):
    """
    Load class labels from file.
    Args:
        labels_path(str): List of classes file path
    Returns:
        List of classes
    """
    with open(labels_path, "r") as fp:
        data = fp.readlines()
    data = [dt.strip() for dt in data]
    return data


def load_class_imbalances(class_imbalances_path):
    """
    Load class imbalances from json file.
    Args:
        class_imbalances_path(str): Class imbalances path
    Returns:
        Dictionary of class imbalances
    """
    with open(class_imbalances_path, "r") as fp:
        class_imbalances = json.load(fp)
    return class_imbalances


def get_best_model_path(tuning_job_result):
    """Gets model path in the S3 from the tuning job outputs
    Args:
        tuning_job_result(object): Hyperparameter tuning result
    Returns:
        str: Best model path from the tuning jobs
    """
    best_job = tuning_job_result.get("BestTrainingJob", None)
    job_name = best_job["TrainingJobName"]
    model_name = job_name + "-model"
    info = smclient.describe_training_job(TrainingJobName=job_name)
    model_path = info["ModelArtifacts"]["S3ModelArtifacts"]
    return model_path


def get_tuner_status_and_result_until_completion(
    tuner, num_features, target, sleep_time=60
):
    """Print results of running tuner on a regular interval until completion
    Args:
        tuner(Object): The running Hyperparameter tuner object
        num_features(int): Total number of features
        target(str): Target string
    Returns:
        None
    """
    print("Training for #features={} and class={}...".format(num_features, target))
    while True:
        tuning_job_result = smclient.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuner.latest_tuning_job.job_name
        )
        job_count = tuning_job_result["TrainingJobStatusCounters"]["Completed"]
        status = tuning_job_result["HyperParameterTuningJobStatus"]

        auc_value = None
        if tuning_job_result.get("BestTrainingJob", None):
            best_job = tuning_job_result["BestTrainingJob"]
            metric = best_job["FinalHyperParameterTuningJobObjectiveMetric"][
                "MetricName"
            ]
            auc_value = best_job["FinalHyperParameterTuningJobObjectiveMetric"]["Value"]
            auc_value = round(auc_value, 4)
            print("Total jobs completed: {}".format(job_count))
            print("Metric: {}".format(metric))
            print("Best AUC: {}".format(auc_value))
        else:
            print("-")

        if status == "Completed":
            model_path = get_best_model_path(tuning_job_result)
            print(
                "Training for #features={} class={} completed!".format(
                    num_features, target
                )
            )
            print("==================================")
            return auc_value, model_path

        time.sleep(sleep_time)


if __name__ == "__main__":
    TRAIN_RESULTS_PATH = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/ae/1000/training/train_results.csv"
    TRAIN_TEMP_RESULTS_DIR = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/ae/1000/training/temp_results/"
    LABELS_PATH = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/ae/1000/raw/labels.txt"
    CLASS_IMBALANCE_PATH = "/home/ec2-user/SageMaker/CMSAI/modeling/tes/data/final-global/ae/1000/raw/class_imbalances.json"

    # Bucket where the trained model is stored
    BUCKET = "cmsai-mrk-amzn"
    # Directory prefix where the model training outputs is saved
    now = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    DATA_PREFIX = "CSVModelInputs/Tes/models/ae/final-global/data"
    MODEL_PREFIX = "CSVModelInputs/Tes/models/ae/final-global/xgboost/{}".format(now)

    # Number of features and labels used for training
    NUM_FEATURES_LIST = [200]  # [100, 200, 300]

    ###Algorithm config
    ALGORITHM = "xgboost"
    REPO_VERSION = "1.2-1"

    ###Hyperparameter tuning config
    TRAIN_INSTANCE_TYPE = "ml.m4.16xlarge"
    TRAIN_INSTANCE_COUNT = 2
    MAX_PARALLEL_JOBS = 4
    MAX_TRAIN_JOBS = 8  # 20

    EVALUATION_METRIC = "auc"
    OBJECTIVE = "binary:logistic"
    OBJECTIVE_METRIC_NAME = "validation:auc"

    # Update hyperparameter ranges
    HYPERPARAMETER_RANGES = {
        "eta": ContinuousParameter(0, 1),
        "alpha": ContinuousParameter(0, 2),
        "max_depth": IntegerParameter(1, 10),
    }

    #     HYPERPARAMETER_RANGES = {'eta': ContinuousParameter(0.1, 0.5),
    #                            'alpha': ContinuousParameter(0, 2),
    #                            'max_depth': IntegerParameter(1, 10),
    #                            'gamma': ContinuousParameter(0, 5),
    #                            'num_round': IntegerParameter(200, 500),
    #                            'colsample_bylevel': ContinuousParameter(0.1, 1.0),
    #                            'colsample_bynode': ContinuousParameter(0.1, 1.0),
    #                            'colsample_bytree': ContinuousParameter(0.5, 1.0),
    #                            'lambda': ContinuousParameter(0, 1000),
    #                            'max_delta_step': IntegerParameter(0, 10),
    #                            'min_child_weight': ContinuousParameter(0, 120),
    #                            'subsample': ContinuousParameter(0.5, 1.0),
    #                            }

    ### SageMaker Initialization
    region = boto3.Session().region_name
    role = sagemaker.get_execution_role()
    smclient = boto3.Session().client("sagemaker")

    sess = sagemaker.Session()

    container = retrieve(ALGORITHM, region, version=REPO_VERSION)

    training_results = []
    s3_resource = boto3.Session().resource("s3")

    if not os.path.exists(TRAIN_TEMP_RESULTS_DIR):
        os.makedirs(TRAIN_TEMP_RESULTS_DIR)
    # Load class imbalances
    class_imbalances = load_class_imbalances(CLASS_IMBALANCE_PATH)
    # Load list of labels
    labels = load_labels(LABELS_PATH)
    for num_features in NUM_FEATURES_LIST:
        for label in labels:
            start = time.time()
            print(
                "Training for num_features={}, label={}...".format(num_features, label)
            )
            # Prepare the input train & validation data path
            s3_input_train = sagemaker.inputs.TrainingInput(
                s3_data="s3://{}/{}/{}/{}/train".format(
                    BUCKET, DATA_PREFIX, num_features, label
                ),
                content_type="csv",
            )
            s3_input_validation = sagemaker.inputs.TrainingInput(
                s3_data="s3://{}/{}/{}/{}/val".format(
                    BUCKET, DATA_PREFIX, num_features, label
                ),
                content_type="csv",
            )

            # Class Imbalance
            imb = class_imbalances[label]
            scale_pos_weight = float(imb[0]) / imb[1]  # negative/positive

            xgb = sagemaker.estimator.Estimator(
                container,
                role,
                instance_count=TRAIN_INSTANCE_COUNT,
                instance_type=TRAIN_INSTANCE_TYPE,
                output_path="s3://{}/{}/{}/{}/output".format(
                    BUCKET, MODEL_PREFIX, num_features, label
                ),
                sagemaker_session=sess,
            )

            xgb.set_hyperparameters(
                eval_metric=EVALUATION_METRIC,
                objective=OBJECTIVE,
                scale_pos_weight=scale_pos_weight,  # For class imbalance
                num_round=200,
                rate_drop=0.3,
                max_depth=5,
                subsample=0.8,
                gamma=2,
                eta=0.2,
            )

            tuner = HyperparameterTuner(
                xgb,
                OBJECTIVE_METRIC_NAME,
                HYPERPARAMETER_RANGES,
                max_jobs=MAX_TRAIN_JOBS,
                max_parallel_jobs=MAX_PARALLEL_JOBS,
            )

            # Training the HPO tuner
            tuner.fit(
                {"train": s3_input_train, "validation": s3_input_validation},
                include_cls_metadata=False,
            )

            # Get the hyperparameter tuner status at regular interval
            val_auc, best_model_path = get_tuner_status_and_result_until_completion(
                tuner, num_features, label
            )

            result = [label, num_features, val_auc, best_model_path]
            training_results.append(result)

            result = [str(res) for res in result]
            temp_result_path = os.path.join(
                TRAIN_TEMP_RESULTS_DIR, "{}_{}.txt".format(num_features, label)
            )
            with open(temp_result_path, "w") as fp:
                fp.write(",".join(result))

            print(
                "Success! Total training time={} mins.".format(
                    (time.time() - start) / 60.0
                )
            )
    # Save the results to file
    df_results = pd.DataFrame(
        training_results,
        columns=["class", "num_features", "val_auc", "best_model_path"],
    )
    df_results.to_csv(TRAIN_RESULTS_PATH, index=False)
    print("ALL SUCCESS!")

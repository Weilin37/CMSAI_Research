from collections import Counter

import xgboost
import sagemaker
import boto3
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.image_uris import retrieve
from urllib.parse import urlparse
import tarfile
import pickle
import shutil
import os
import pandas as pd

import shap
import xgboost as xgb

import sagemaker
import boto3
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.image_uris import retrieve


def get_valid_tokens(tokens):
    """Get all tokens except <pad> and <unk>"""
    my_tokens = []
    for key, val in tokens.items():
        if val >= 2:
            my_tokens.append(key)
    my_tokens
    return my_tokens


def get_one_hot(df, tokens_list, seq_len, target_colname, use_freq=False):
    """Compute one hot encoding of the dataset.
    use_freq: Whether to use frequency of features or one-hot encoding
    """

    def _get_one_hot_row(row, tokens_list):
        """Get one-hot encoding for a single row."""
        if use_freq:
            counter = Counter(row.tolist())
            one_hot = [counter[token] for token in tokens_list]
        else:
            features = set(row.tolist())
            one_hot = [int(ft in features) for ft in tokens_list]
        return one_hot

    feature_colnames = [str(x) for x in range(seq_len - 1, -1, -1)]
    df_x = df.loc[:, feature_colnames]
    df_y = df.loc[:, [target_colname]]

    df_x = df.apply(_get_one_hot_row, axis=1, args=(tokens_list,))
    df_x = pd.DataFrame(df_x.tolist(), columns=tokens_list)

    df0 = pd.concat([df_x, df_y], axis=1)
    df0.set_index(pd.Index(df.patient_id), inplace=True)
    return df0


def prepare_data(
    input_path, out_one_hot_path, out_data_path, seq_len, target_colname, tokens, s3_dir, use_freq=False
):
    """Prepare data for xgboost training."""
    df = pd.read_csv(input_path)
    df0 = get_one_hot(df, tokens, seq_len, target_colname, use_freq=use_freq)

    if not os.path.isdir(os.path.split(out_one_hot_path)[0]):
        os.makedirs(os.path.split(out_one_hot_path)[0])
    if not os.path.isdir(os.path.split(out_data_path)[0]):
        os.makedirs(os.path.split(out_data_path)[0])

    df0.to_csv(out_one_hot_path)

    columns = df0.columns.tolist()
    columns = [columns[-1]] + columns[:-1]
    df0[columns].to_csv(out_data_path, index=False, header=None)

    fname = os.path.basename(out_data_path)
    s3_path = os.path.join(s3_dir, fname)
    command = "aws s3 cp {} {}".format(out_data_path, s3_path)
    os.system(command)
    print("Sucess!")


def train_hpo(
    hyperparameter_ranges,
    container,
    execution_role,
    instance_count,
    instance_type,
    output_path,
    sagemaker_session,
    eval_metric,
    objective,
    objective_metric_name,
    max_train_jobs,
    max_parallel_jobs,
    scale_pos_weight,
    data_channels,
):
    """
    Train a model based on a given data fold and HPO training job summary job.
    Args:
        hyperparameter_ranges(dict): Dictionary of all hyperparameter ranges
        container(Object): Xgboost model docker container
        execution_role(Object): Role to enable execution of HPO job
        instance_count(int): # of instances for training job
        instance_type(str): Instance type
        output_path(str): Output path
        sagemaker_session(Object): SageMaker session
        eval_metric(str): Evaluation metric
        objective(str): Objective function name
        objective_metric_name(str): Objective function metric name
        max_train_jobs(int): Max number of training jobs to run
        max_parallel_jobs(int): Max number of jobs to run in parallel
        scale_pos_weight: Class imbalance weight scale
        data_channels(dict): Dictionary of data channels to be used for training
    Returns:
        Tuner object
    """
    xgb_model = sagemaker.estimator.Estimator(
        container,
        execution_role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_path,
        sagemaker_session=sagemaker_session,
    )

    xgb_model.set_hyperparameters(
        eval_metric=eval_metric,
        objective=objective,
        scale_pos_weight=scale_pos_weight,  # For class imbalance
        num_round=200,
        rate_drop=0.3,
        max_depth=5,
        subsample=0.8,
        gamma=2,
        eta=0.2,
    )

    tuner = HyperparameterTuner(
        xgb_model,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=max_train_jobs,
        max_parallel_jobs=max_parallel_jobs,
    )

    tuner.fit(inputs=data_channels)
    return tuner


def get_tuner_status_and_result_until_completion(
    tuner, num_features, target, sleep_time=60
):
    """Print results of running tuner on a regular interval until completion
    Args:
        tuner(Object): The running Hyperparameter tuner object
        target(str): Target string
    Returns:
        None
    """
    smclient = boto3.Session().client("sagemaker")
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
            return auc_value, model_path

        time.sleep(sleep_time)


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
    smclient = boto3.Session().client("sagemaker")
    info = smclient.describe_training_job(TrainingJobName=job_name)
    model_path = info["ModelArtifacts"]["S3ModelArtifacts"]
    return model_path


def copy_model_from_s3(s3_model_path, local_model_dir):
    """Copy model from s3 to local
    Args:
        s3_model_path(str): S3 path where the model gz is saved
    Returns:
        Destination model path
    """
    client = boto3.client("s3")
    o = urlparse(s3_model_path)
    bucket = o.netloc
    key = o.path
    key = key.lstrip("/")
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)
    fname = os.path.basename(s3_model_path)
    output_path = os.path.join(local_model_dir, fname)

    client.download_file(bucket, key, output_path)

    return output_path


def load_model(gz_model_path, remove=True):
    """
    Loads xgboost trained model from disk
    Args:
        gz_model_path(str): Compressed Model path
    Returns:
        xgboost: Xgboost model object
    """
    model_dir = os.path.dirname(gz_model_path)
    model_path = os.path.join(model_dir, "xgboost-model")

    tar = tarfile.open(gz_model_path, "r:gz")
    tar.extractall(model_dir)
    tar.close()

    # Load Model
    model = pickle.load(open(model_path, "rb"))

    # Remove the local copy of the model files if needed
    if remove:
        shutil.rmtree(model_dir)
    else:
        os.remove(model_path) #Remove only the extracted file

    return model

def get_valid_tokens(df, seq_len):
    """Get list of tokens."""
    feature_cols = [str(i) for i in range(seq_len-1, -1, -1)]
    tokens = list(set(df[feature_cols].values.flatten().tolist()))
    pad = '<pad>'
    if pad in tokens:
        tokens.remove('<pad>')
    return tokens


def get_best_model_info(df):
    """Get best model path based on its Intersection Sim index."""
    best_idx = df[["val_Intersection_Sim"]].idxmax().tolist()[0]
    best_df_row = df.iloc[best_idx]
    return best_df_row

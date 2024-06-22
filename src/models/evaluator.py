import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "optuna", "sagemaker", "boto3"]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "mlflow==2.13.2",
        "sagemaker-mlflow==0.1.0",
    ]
)

import argparse
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import tarfile
import logging
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import boto3
import time
import mlflow
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup


class ModelEvaluator:
    def __init__(
        self,
        mode,
        input_path,
        dataset_sizes_path,
        data_version,
        target_column,
        columns_to_drop,
        model_path,
        output_path,
        feature_group_name,
        region,
        bucket_name,
        tracking_server_arn,
        experiment_name,
    ):
        self.mode = mode
        self.input_path = input_path
        self.data_version = data_version
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop.split(",") if columns_to_drop else []
        self.columns_to_drop.append(target_column)
        self.model_path = model_path
        self.output_path = output_path
        self.tracking_server_arn = tracking_server_arn
        self.experiment_name = experiment_name

        with open(dataset_sizes_path, "r") as f:
            self.dataset_sizes = json.load(f)

        if self.mode == "feature_store":
            if not feature_group_name or not bucket_name or not region:
                raise ValueError(
                    "feature_group_name, bucket_name and region are required in feature_store mode"
                )
            sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
            self.feature_group = FeatureGroup(
                name=feature_group_name, sagemaker_session=sagemaker_session
            )
            self.bucket_name = bucket_name

    def load_data(self):
        if self.mode == "feature_store":
            test_df = self._load_data_from_feature_store()
        elif self.mode == "local":
            test_df = pd.read_csv(os.path.join(self.input_path, "test.csv"))

        self.X_test = test_df.drop(columns=self.columns_to_drop)
        self.y_test = test_df[self.target_column]

        mlflow.log_params(
            {
                "test_dataset_size": len(self.y_test),
            },
            run_id=self.run_id,
        )

        self.close_prices = test_df["close"].values
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def _get_record_count(self, record_type):
        fs_query = self.feature_group.athena_query()
        query_string = f"""
            SELECT COUNT(*) 
            FROM "{fs_query.table_name}" 
            WHERE version = '{self.data_version}' AND type = '{record_type}'
        """
        output_location = f"s3://{self.bucket_name}/tmp/offline-store/query_results/"

        fs_query.run(query_string=query_string, output_location=output_location)
        fs_query.wait()
        fs_df = fs_query.as_dataframe()

        return fs_df.iat[0, 0]

    def _wait_until_ingestion_complete(self):
        test_count = self._get_record_count("test")

        while test_count < self.dataset_sizes["test_size"]:
            print("Waiting for ingestion to complete...")
            time.sleep(30)
            test_count = self._get_record_count("test")

    def _load_data_from_feature_store(self):
        self._wait_until_ingestion_complete()

        fs_query = self.feature_group.athena_query()

        query_string = f"""
            SELECT *
            FROM "{fs_query.table_name}"
            WHERE version = '{self.data_version}' AND type = 'test'
        """

        output_location = f"s3://{self.bucket_name}/tmp/offline-store/query_results/"

        fs_query.run(
            query_string=query_string,
            output_location=f"{output_location}/test",
        )

        fs_query.wait()
        return fs_query.as_dataframe()

    def load_model(self):
        tarfile_path = f"{self.model_path}/model.tar.gz"
        model_file_path = f"{self.model_path}/model.xgb"

        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(model_file_path))

        self.model = xgb.Booster()
        self.model.load_model(model_file_path)
        mlflow.log_artifact(self.model_path, "model.xgb", run_id=self.run_id)

    def evaluate_model(self):
        y_pred_test = self.model.predict(self.dtest)
        y_pred_test_binary = (y_pred_test > 0.5).astype(int)

        self.test_accuracy = accuracy_score(self.y_test, y_pred_test_binary)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred_test_binary)
        self.class_report = classification_report(self.y_test, y_pred_test_binary)

        self.cumulative_reward = self.compute_cumulative_reward(
            y_pred_test_binary, self.close_prices
        )
        self.cumulative_return = self.compute_cumulative_return(
            y_pred_test_binary, self.close_prices
        )

    def compute_cumulative_reward(self, y_pred, close_prices):
        rewards = []
        for i in range(0, len(close_prices) - 3):
            if y_pred[i] == 1:
                rewards.append(close_prices[i + 3] - close_prices[i])
            else:
                rewards.append(close_prices[i] - close_prices[i + 3])
        return np.sum(rewards)

    def compute_cumulative_return(self, y_pred, close_prices):
        rewards = []
        for i in range(0, len(close_prices) - 3):
            if y_pred[i] == 1:
                rewards.append(
                    (close_prices[i + 3] - close_prices[i]) / close_prices[i]
                )
            else:
                rewards.append(
                    (close_prices[i] - close_prices[i + 3]) / close_prices[i]
                )
        return np.sum(rewards)

    def save_evaluation(self):
        os.makedirs(self.output_path, exist_ok=True)
        evaluation_report_path = os.path.join(
            self.output_path, "evaluation_report.json"
        )

        evaluation_report = {
            "cumulative_reward": self.cumulative_reward,
            "cumulative_return": self.cumulative_return,
            "test_accuracy": self.test_accuracy * 100,
            "confusion_matrix": self.conf_matrix.tolist(),
        }

        mlflow.log_metrics(
            {
                "cumulative_reward": self.cumulative_reward,
                "cumulative_return": self.cumulative_return,
                "test_accuracy": self.test_accuracy,
            },
            run_id=self.run_id,
        )

        with open(evaluation_report_path, "w") as f:
            json.dump(evaluation_report, f, indent=4)

        print(f"Evaluation report saved to {evaluation_report_path}")

    def run(self):
        mlflow.set_tracking_uri(self.tracking_server_arn)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_name=sagemaker.utils.name_from_base(f"{self.experiment_name}-job")
        ) as run:
            self.run_id = run.info.run_id

            self.load_data()
            self.load_model()
            self.evaluate_model()
            self.save_evaluation()

            mlflow.end_run(status="FINISHED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("model_evaluator")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode of the script. Possible values: 'feature_store', 'local'.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        help="Path to the input test data (used in local mode)",
    )
    parser.add_argument(
        "--dataset_sizes_path",
        type=str,
        required=False,
        help="Path to the json file containing dataset sizes (used in feature store mode)",
    )
    parser.add_argument(
        "--data_version",
        type=str,
        required=True,
        help="Version of the input test data",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column",
    )
    parser.add_argument(
        "--columns_to_drop",
        type=str,
        required=False,
        help="Columns to drop from the input data, separated by commas",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model tar.gz file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--feature_group_name",
        type=str,
        required=False,
        help="Name of the feature group in Feature Store.",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        help="AWS region of Sagemaker Session.",
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        required=False,
        help="S3 bucket name (used in feature store mode)",
    )
    parser.add_argument(
        "--tracking_server_arn",
        type=str,
        required=True,
        help="MLFlow tracking server to track experiment",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="MLFlow experiment name",
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        mode=args.mode,
        input_path=args.input_path,
        dataset_sizes_path=args.dataset_sizes_path,
        data_version=args.data_version,
        target_column=args.target_column,
        columns_to_drop=args.columns_to_drop,
        model_path=args.model_path,
        output_path=args.output_path,
        feature_group_name=args.feature_group_name,
        region=args.region,
        bucket_name=args.bucket_name,
        tracking_server_arn=args.tracking_server_arn,
        experiment_name=args.experiment_name,
    )

    evaluator.run()

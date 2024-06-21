import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "optuna", "sagemaker", "boto3"]
)

import argparse
import pandas as pd
import xgboost as xgb
from optuna import create_study
from sklearn.metrics import accuracy_score
import os
import tarfile
import logging
import json
import time
import boto3

import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup


class ModelTrainer:
    def __init__(
        self,
        mode,
        input_path,
        data_version,
        target_column,
        columns_to_drop,
        model_output_path,
        num_trials,
        feature_group_name,
        dataset_sizes_path=None,
        bucket_name=None,
        region=None,
    ):
        self.target_column = target_column
        self.model_output_path = model_output_path
        self.num_trials = num_trials
        self.mode = mode
        self.columns_to_drop = columns_to_drop.split(",") if columns_to_drop else []
        self.columns_to_drop.append(target_column)
        self.feature_group_name = feature_group_name
        self.bucket_name = bucket_name
        self.data_version = data_version
        self.region = region

        if self.mode == "feature_store":
            if (
                not self.feature_group_name
                or not self.bucket_name
                or not dataset_sizes_path
                or not region
            ):
                raise ValueError(
                    "feature_group_name, bucket_name, dataset_sizes_path and \
                    region are required in feature_store mode"
                )

            with open(self.dataset_sizes_path, "r") as f:
                self.dataset_sizes = json.load(f)
        elif self.mode == "local":
            self.data_path = os.path.join(input_path, "data.csv")

    def load_data(self):
        if self.mode == "local":
            df = pd.read_csv(self.data_path)
            train_df = df[df["type"] == "train"]
            validation_df = df[df["type"] == "validation"]
        elif self.mode == "feature_store":
            train_df, validation_df = self._load_data_from_feature_store()

        self.y_train = train_df[self.target_column]
        self.X_train = train_df.drop(columns=self.columns_to_drop)

        self.y_validation = validation_df[self.target_column]
        self.X_validation = validation_df.drop(columns=self.columns_to_drop)

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dvalidation = xgb.DMatrix(self.X_validation, label=self.y_validation)

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
        train_count = self._get_record_count("train")
        validation_count = self._get_record_count("validation")

        while (
            train_count < self.dataset_sizes["train"]
            or validation_count < self.dataset_sizes["validation"]
        ):
            print("Waiting for ingestion to complete...")
            time.sleep(30)
            train_count = self._get_record_count("train")
            validation_count = self._get_record_count("validation")

    def _load_data_from_feature_store(self):
        sagemaker_session = sagemaker.Session(boto3.Session(region_name=self.region))
        self.feature_group = FeatureGroup(
            name=self.feature_group_name, sagemaker_session=sagemaker_session
        )

        self._wait_until_ingestion_complete()

        fs_train_query = self.feature_group.athena_query()
        fs_validation_query = self.feature_group.athena_query()

        train_query_string = f"""
            SELECT *
            FROM "{fs_train_query.table_name}"
            WHERE version = '{self.data_version}' AND type = 'train'
        """
        validation_query_string = f"""
            SELECT *
            FROM "{fs_validation_query.table_name}"
            WHERE version = '{self.data_version}' AND type = 'validation'
        """

        output_location = f"s3://{self.bucket_name}/tmp/offline-store/query_results/"

        fs_train_query.run(
            query_string=train_query_string,
            output_location=f"{output_location}/train",
        )
        fs_validation_query.run(
            query_string=validation_query_string,
            output_location=f"{output_location}/validation",
        )

        fs_train_query.wait()
        fs_validation_query.wait()

        train_df = fs_train_query.as_dataframe()
        validation_df = fs_validation_query.as_dataframe()

        return train_df, validation_df

    def objective(self, trial):
        params = {
            "objective": "binary:logistic",
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "eval_metric": "logloss",
        }
        num_boost_round = trial.suggest_int("num_boost_round", 50, 200)

        evals = [(self.dtrain, "train"), (self.dvalidation, "eval")]

        bst = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        y_pred_validation = bst.predict(self.dvalidation)
        y_pred_validation_binary = (y_pred_validation > 0.5).astype(int)

        validation_accuracy = accuracy_score(
            self.y_validation, y_pred_validation_binary
        )

        return validation_accuracy

    def optimize_hyperparameters(self):
        study = create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial), n_trials=self.num_trials)

        self.best_params = study.best_params

    def train_model(self):
        params = {
            "objective": "binary:logistic",
            "max_depth": self.best_params["max_depth"],
            "learning_rate": self.best_params["learning_rate"],
            "eval_metric": "logloss",
        }

        evals = [(self.dtrain, "train"), (self.dvalidation, "eval")]

        self.bst = xgb.train(
            params,
            self.dtrain,
            num_boost_round=self.best_params["num_boost_round"],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

    def print_validation_accuracy(self):
        y_pred_validation = self.bst.predict(self.dvalidation)

        y_pred_validation_binary = (y_pred_validation > 0.5).astype(int)

        validation_accuracy = accuracy_score(
            self.y_validation, y_pred_validation_binary
        )

        print(f"Validation Accuracy: {validation_accuracy*100:.2f}%")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        self.bst.save_model(self.model_output_path + ".xgb")
        print(f"Model saved to {self.model_output_path}")

    def run(self):
        self.load_data()
        self.optimize_hyperparameters()
        self.train_model()
        self.print_validation_accuracy()
        self.save_model()


if __name__ == "__main__":
    logger = logging.getLogger("optuna")
    logger.setLevel(logging.WARNING)

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
        help="Path to the input data (used in local mode)",
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
        help="Version of the input data",
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
        "--model_output_path",
        type=str,
        required=True,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num_trials", type=int, required=True, help="Number of Optuna trials"
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

    args = parser.parse_args()

    trainer = ModelTrainer(
        mode=args.mode,
        input_path=args.input_path,
        dataset_sizes_path=args.dataset_sizes_path,
        data_version=args.data_version,
        target_column=args.target_column,
        columns_to_drop=args.columns_to_drop,
        model_output_path=args.model_output_path,
        num_trials=args.num_trials,
        feature_group_name=args.feature_group_name,
        region=args.region,
    )

    trainer.run()

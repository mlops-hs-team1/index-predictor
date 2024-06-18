import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])

import argparse
import pandas as pd
import xgboost as xgb
from optuna import create_study
from sklearn.metrics import accuracy_score
import os
import tarfile
import logging


class ModelTrainer:
    def __init__(
        self,
        input_path,
        data_version,
        target_column,
        model_output_path,
        num_trials,
    ):
        self.train_data_path = os.path.join(input_path, f"train-{data_version}.csv")
        self.validation_data_path = os.path.join(
            input_path, f"validation-{data_version}.csv"
        )
        self.target_column = target_column
        self.model_output_path = model_output_path
        self.num_trials = num_trials

    def load_data(self):
        train_df = pd.read_csv(self.train_data_path)
        validation_df = pd.read_csv(self.validation_data_path)

        self.X_train = train_df.drop(columns=[self.target_column])
        self.y_train = train_df[self.target_column]

        self.X_validation = validation_df.drop(columns=[self.target_column])
        self.y_validation = validation_df[self.target_column]

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dvalidation = xgb.DMatrix(self.X_validation, label=self.y_validation)

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
        with tarfile.open(self.model_output_path + ".tar.gz", "w:gz") as tar:
            tar.add(
                self.model_output_path + ".xgb",
                arcname=os.path.basename(self.model_output_path + ".xgb"),
            )
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
        "--input-path",
        type=str,
        required=True,
        help="Path to the input data",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        required=True,
        help="Version of the input data",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        required=True,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num-trials", type=int, required=True, help="Number of Optuna trials"
    )

    args = parser.parse_args()

    trainer = ModelTrainer(
        input_path=args.input_path,
        data_version=args.data_version,
        target_column=args.target_column,
        model_output_path=args.model_output_path,
        num_trials=args.num_trials,
    )

    trainer.run()

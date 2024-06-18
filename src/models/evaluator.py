import argparse
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import tarfile
import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ModelEvaluator:
    def __init__(
        self,
        input_path,
        data_version,
        target_column,
        model_path,
        output_path,
    ):
        self.test_data_path = os.path.join(input_path, f"test-{data_version}.csv")
        self.target_column = target_column
        self.model_path = model_path
        self.output_path = output_path

    def load_data(self):
        test_df = pd.read_csv(self.test_data_path)
        self.X_test = test_df.drop(columns=[self.target_column])
        self.y_test = test_df[self.target_column]
        self.close_prices = test_df["Close"].values
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def load_model(self):
        tarfile_path = self.model_path + ".tar.gz"
        model_file_path = self.model_path + ".xgb"

        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(model_file_path))

        self.model = xgb.Booster()
        self.model.load_model(model_file_path)

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
        evaluation_report = os.path.join(self.output_path, "evaluation_report.txt")
        with open(evaluation_report, "w") as f:
            f.write(f"\n\nCumulative Reward: {self.cumulative_reward}\n")
            f.write(f"Cumulative Return: {self.cumulative_return}\n")
            f.write(f"Test Accuracy: {self.test_accuracy*100:.2f}%\n")
            f.write("Confusion Matrix:\n")
            f.write(str(self.conf_matrix))
            f.write("\n\nClassification Report:\n")
            f.write(self.class_report)
        print(f"Evaluation report saved to {evaluation_report}")

    def run(self):
        self.load_data()
        self.load_model()
        self.evaluate_model()
        self.save_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("model_evaluator")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input test data",
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

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        input_path=args.input_path,
        data_version=args.data_version,
        target_column=args.target_column,
        model_path=args.model_path,
        output_path=args.output_path,
    )

    evaluator.run()

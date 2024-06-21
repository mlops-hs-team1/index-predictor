import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker", "boto3"])

import argparse
import pandas as pd
import os
import json

import sagemaker
import boto3
from sagemaker.feature_store.feature_group import FeatureGroup


class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, raw_data_filename):
        df = pd.read_csv(raw_data_filename)
        return df

    def drop_columns(self, df, columns=["Adj Close"]):
        df.drop(columns=columns, inplace=True)
        return df

    def sort_by_datetime(self, df):
        return df.sort_values(by=["Datetime"])

    def extract_date_features(self, df):
        df["DayOfWeek"] = pd.to_datetime(df["Datetime"]).dt.dayofweek
        df["Hour"] = pd.to_datetime(df["Datetime"]).dt.hour
        df["Minute"] = pd.to_datetime(df["Datetime"]).dt.minute
        return df

    def one_hot_encode_day_of_week(self, df, max_day_of_week=4):
        for i in range(max_day_of_week + 1):
            df[f"DayOfWeek_{i}"] = (df["DayOfWeek"] == i).astype(int)
        df.drop(columns=["DayOfWeek"], inplace=True)
        return df

    def split_data(self, df, validation_size=390, test_size=390):
        train_df = df.iloc[: -validation_size - test_size]
        validation_df = df.iloc[-validation_size - test_size : -test_size]
        test_df = df.iloc[-test_size:]
        return train_df, validation_df, test_df

    def convert_datetime_to_iso_8601(self, df):
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return df

    def prepare_data(self, df, lag):
        df = df.copy()

        # Prepare lagged features
        lagged_data = {}
        for i in range(1, lag + 1):
            for col in ["Open", "Close", "High", "Low", "Volume"]:
                lagged_data[f"{col}_lag_{i}"] = df[col].shift(i)

        lagged_df = pd.DataFrame(lagged_data)
        df = pd.concat([df, lagged_df], axis=1)

        # Create target variable: whether Close price will be higher in 3 minutes
        df["Close_target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)

        df.dropna(inplace=True)
        return df

    def convert_col_name(self, c):
        return c.lower().replace(".", "_").replace("-", "_").rstrip("_")

    def ingest_data(self, df, feature_group_name, region):
        sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
        feature_group = FeatureGroup(
            name=feature_group_name, sagemaker_session=sagemaker_session
        )
        feature_group.ingest(data_frame=df, max_workers=3, wait=True)
        print("Data ingested to feature store")

    def store_dataset_sizes(self, train_size, validation_size, test_size, output_path):
        dataset_sizes = {
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
        }
        with open(os.path.join(output_path, "dataset_sizes.json"), "w") as f:
            json.dump(dataset_sizes, f)
        print(f"Dataset sizes saved at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode of the script. Possible values: 'feature_store', 'local'.",
    )
    parser.add_argument(
        "--raw_data_filename",
        type=str,
        required=True,
        help="Filename of raw training data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where processed data will be saved.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version of the dataset.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = DataProcessor()

    df = processor.load_data(args.raw_data_filename)

    df = processor.drop_columns(df)

    df = processor.sort_by_datetime(df)
    df = processor.extract_date_features(df)
    df = processor.one_hot_encode_day_of_week(df)

    df = processor.convert_datetime_to_iso_8601(df)

    train_df, validation_df, test_df = processor.split_data(df)

    lag = 30
    train_df = processor.prepare_data(train_df, lag)
    validation_df = processor.prepare_data(validation_df, lag)
    test_df = processor.prepare_data(test_df, lag)

    train_df["type"] = ["train"] * train_df.shape[0]
    validation_df["type"] = ["validation"] * validation_df.shape[0]
    test_df["type"] = ["test"] * test_df.shape[0]

    merged_df = pd.concat([train_df, validation_df, test_df], axis=0)
    merged_df["version"] = [args.version] * merged_df.shape[0]

    merged_df = merged_df.rename(columns=processor.convert_col_name)

    if args.mode == "local":
        merged_df.to_csv(os.path.join(args.output_path, "data.csv"), index=False)
    elif args.mode == "feature_store":
        if not args.feature_group_name:
            raise ValueError(
                "Please provide the name of the feature group in Feature Store."
            )
        processor.ingest_data(merged_df, args.feature_group_name, args.region)

    processor.store_dataset_sizes(
        train_size=train_df.shape[0],
        validation_size=validation_df.shape[0],
        test_size=test_df.shape[0],
        output_path=args.output_path,
    )

    print(f"Data processing completed.")

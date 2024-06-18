import argparse
import pandas as pd
import os


class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, raw_data_filename):
        df = pd.read_csv(raw_data_filename)
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

    def store_data(self, df, output_path, type, version):
        os.makedirs(args.output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{type}-{version}.csv")
        df.to_csv(output_path, index=False)
        print(f"{type} data saved at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = DataProcessor()

    df = processor.load_data(args.raw_data_filename)

    df = processor.sort_by_datetime(df)
    df = processor.extract_date_features(df)
    df = processor.one_hot_encode_day_of_week(df)

    train_df, validation_df, test_df = processor.split_data(df)

    lag = 30
    train_df = processor.prepare_data(train_df, lag)
    validation_df = processor.prepare_data(validation_df, lag)
    test_df = processor.prepare_data(test_df, lag)

    processor.store_data(train_df, args.output_path, "train", args.version)
    processor.store_data(validation_df, args.output_path, "validation", args.version)
    processor.store_data(test_df, args.output_path, "test", args.version)

    print(f"Data processing completed.")

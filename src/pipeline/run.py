import os
import sagemaker
import time
import argparse
from sagemaker.workflow.pipeline import Pipeline

if __name__ == "__main__":
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=timestamp)

    args = parser.parse_args()

    sagemaker_session = sagemaker.Session()

    pipeline = Pipeline(
        name="index-predictor-pipeline", sagemaker_session=sagemaker_session
    )

    execution = pipeline.start(
        parameters={
            "Version": args.version,
        },
        execution_display_name=f"Execution-{timestamp}",
        execution_description=f"Pipeline execution for version {args.version}",
    )

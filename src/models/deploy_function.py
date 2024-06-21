import sagemaker
import json
import boto3
import os


def lambda_handler(event, context):
    bucket_name = os.environ["BUCKET_NAME"]
    model_path = os.environ["MODEL_PATH"]
    role = os.environ["ROLE"]
    region = os.environ["REGION"]

    sagemaker_session = boto3.Session().client("sagemaker", region_name=region)

    xgboost_container = boto3.client("sagemaker").image_uris.retrieve(
        "xgboost", region, version="1.3-1"
    )

    model = sagemaker.model.Model(
        image_uri=xgboost_container,
        model_data=f"s3://{bucket_name}/{model_path}",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    predictor = model.deploy(initial_instance_count=1, instance_type="ml.t2.large")

    return {"statusCode": 200, "body": json.dumps("Model deployed successfully!")}

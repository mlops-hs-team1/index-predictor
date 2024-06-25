import logging
import boto3
import sagemaker
import argparse
from botocore.exceptions import ClientError
from datetime import datetime


def get_approved_package(model_package_group_name, sm_client, logger: logging.Logger):
    """Gets the latest approved model package for a model package group."""
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(
                "Getting more packages for token: {}".format(response["NextToken"])
            )
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        if len(approved_packages) == 0:
            error_message = f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            logger.error(error_message)
            raise Exception(error_message)

        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(
            f"Identified the latest approved model package: {model_package_arn}"
        )
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def delete_existing_endpoint(
    endpoint_name,
    sm_client,
):
    """Deletes the existing endpoint if it exists."""
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        if response["EndpointStatus"] != "Deleting":
            print(f"Deleting existing endpoint: {endpoint_name}")
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            sm_client.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
            print(f"Deleted existing endpoint: {endpoint_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint {endpoint_name} does not exist, no need to delete.")
        else:
            error_message = e.response["Error"]["Message"]
            print(error_message)
            raise Exception(error_message)


def deploy_model_endpoint(
    model_package_arn,
    version,
    endpoint_name,
    instance_type,
    initial_instance_count,
    sm_client,
):
    """Deploys the model endpoint using the latest approved package."""
    try:
        create_model_response = sm_client.create_model(
            ModelName=f"{endpoint_name}-{version}",
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=sagemaker.get_execution_role(),
        )
        print(f"Model creation response: {create_model_response}")

        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=f"{endpoint_name}-{version}",
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": f"{endpoint_name}-{version}",
                    "InstanceType": instance_type,
                    "InitialInstanceCount": initial_instance_count,
                },
            ],
        )
        print(f"Endpoint config creation response: {create_endpoint_config_response}")

        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=f"{endpoint_name}-{version}",
        )
        print(f"Endpoint creation response: {create_endpoint_response}")

        sm_client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} is in service")

        return create_endpoint_response
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        print(error_message)
        raise Exception(error_message)


def parse_args():
    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(
        description="Deploy a SageMaker endpoint with the latest approved model package."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=time_now,
        help="Version fo the model and the endpoint condig, default is time now.",
    )
    parser.add_argument(
        "--model-package-group-name",
        type=str,
        default="index-predictor-model-group",
        help="Name of the model package group (default: index-predictor-model-group)",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="index-predictor-endpoint",
        help="Name of the SageMaker endpoint (default: index-predictor-endpoint)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.large",
        help="Instance type for the endpoint (default: ml.m5.large)",
    )
    parser.add_argument(
        "--initial-instance-count",
        type=int,
        default=1,
        help="Initial instance count for the endpoint (default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sm_client = boto3.client("sagemaker")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    model_package_group_name = "index-predictor-model-group"

    model_package_arn = get_approved_package(
        model_package_group_name, sm_client, logger
    )

    delete_existing_endpoint(args.endpoint_name, sm_client)

    deploy_model_endpoint(
        model_package_arn,
        args.version,
        args.endpoint_name,
        args.instance_type,
        args.initial_instance_count,
        sm_client,
    )

import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
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


def extend_config(args, model_package_arn, stage_config):
    """Extend the configuration with additional parameters and tags."""
    if "Parameters" not in stage_config:
        raise Exception("Configuration file must include Parameters section")
    if "Tags" not in stage_config:
        stage_config["Tags"] = {}

    new_params = {
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,
    }
    new_tags = {}

    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


def get_cfn_style_config(stage_config):
    parameters = [
        {"ParameterKey": key, "ParameterValue": value}
        for key, value in stage_config["Parameters"].items()
    ]
    tags = [{"Key": key, "Value": value} for key, value in stage_config["Tags"].items()]
    return parameters, tags


def create_cfn_params_tags_file(config, export_params_file, export_tags_file):
    parameters, tags = get_cfn_style_config(config)
    with open(export_params_file, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(export_tags_file, "w") as f:
        json.dump(tags, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper()
    )
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--import-config", type=str, default="config.json")
    parser.add_argument("--export-config", type=str, default="config-export.json")
    parser.add_argument("--export-params", type=str, default="params-export.json")
    parser.add_argument("--export-tags", type=str, default="tags-export.json")
    parser.add_argument("--export-cfn-params-tags", type=bool, default=False)
    args, _ = parser.parse_known_args()

    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    model_package_arn = get_approved_package(args.model_package_group_name)

    with open(args.import_config, "r") as f:
        config = extend_config(args, model_package_arn, json.load(f))
    logger.debug("Config: {}".format(json.dumps(config, indent=4)))
    with open(args.export_config, "w") as f:
        json.dump(config, f, indent=4)
    if args.export_cfn_params_tags:
        create_cfn_params_tags_file(config, args.export_params, args.export_tags)

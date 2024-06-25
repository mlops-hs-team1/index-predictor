import logging
import sagemaker
from botocore.exceptions import ClientError


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

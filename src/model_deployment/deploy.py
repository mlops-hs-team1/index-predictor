import logging
import boto3
import sagemaker
from get_approved_package import get_approved_package
from botocore.exceptions import ClientError


def deploy_model_endpoint(
    model_package_arn,
    endpoint_name,
    instance_type,
    initial_instance_count,
):
    """Deploys the model endpoint using the latest approved package."""
    try:
        create_model_response = sm_client.create_model(
            ModelName=endpoint_name,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=sagemaker.get_execution_role(),
        )
        logger.info(f"Model creation response: {create_model_response}")

        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": endpoint_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": initial_instance_count,
                },
            ],
        )
        logger.info(
            f"Endpoint config creation response: {create_endpoint_config_response}"
        )

        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )
        logger.info(f"Endpoint creation response: {create_endpoint_response}")

        sm_client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} is in service")

        return create_endpoint_response
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


if __name__ == "__main__":
    sm_client = boto3.client("sagemaker")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    model_package_group_name = "index-predictor-model-group"
    model_package_arn = get_approved_package(
        model_package_group_name, sm_client, logger
    )

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline import PipelineDefinitionConfig
from sagemaker import image_uris
import sagemaker
import boto3
import json
import os
from .steps.collector import get_collector_step
from .steps.processor import get_processor_step
from .steps.trainer import get_trainer_step
from .steps.evaluator import get_evaluator_step
from .steps.register import get_register_step
from .steps.conditional import get_conditional_step


def get_parameters() -> dict:
    version = ParameterString(name="Version", default_value="v1")
    cumulative_return_threshold = ParameterFloat(
        name="CumulativeReturnThreshold",
        default_value=-0.02,
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=0.45,
    )
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large",
    )
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.large",
    )
    model_endpoint_instance_type_param = ParameterString(
        name="ModelEndpointInstanceType",
        default_value="ml.t2.large",
    )
    process_instance_count_param = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    train_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )
    model_endpoint_instance_count_param = ParameterInteger(
        name="ModelEndpointInstanceCount", default_value=1
    )
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    return {
        "version": version,
        "cumulative_return_threshold": cumulative_return_threshold,
        "accuracy_threshold": accuracy_threshold,
        "process_instance_type_param": process_instance_type_param,
        "train_instance_type_param": train_instance_type_param,
        "model_endpoint_instance_type_param": model_endpoint_instance_type_param,
        "process_instance_count_param": process_instance_count_param,
        "train_instance_count_param": train_instance_count_param,
        "model_endpoint_instance_count_param": model_endpoint_instance_count_param,
        "model_approval_status_param": model_approval_status_param,
    }


def get_pipeline(
    session: sagemaker.Session,
    parameters: dict,
    constants: dict,
    sklearn_image_uri: str,
    xgboost_container: str,
):
    pipeline_def_config = PipelineDefinitionConfig(use_custom_job_prefix=True)

    collector_step = get_collector_step(
        project=constants["project"],
        version=parameters["version"],
        bucket_name=constants["bucket_name"],
        process_instance_count_param=parameters["process_instance_count_param"],
        process_instance_type_param=parameters["process_instance_type_param"],
        sklearn_image_uri=sklearn_image_uri,
    )

    processor_step = get_processor_step(
        project=constants["project"],
        version=parameters["version"],
        bucket_name=constants["bucket_name"],
        process_instance_count_param=parameters["process_instance_count_param"],
        process_instance_type_param=parameters["process_instance_type_param"],
        sklearn_image_uri=sklearn_image_uri,
        collection_step=collector_step,
        feature_group_name=constants["feature_group_name"],
        region=constants["region"],
    )

    trainer_step = get_trainer_step(
        xgboost_container=xgboost_container,
        bucket_name=constants["bucket_name"],
        version=parameters["version"],
        project=constants["project"],
        feature_group_name=constants["feature_group_name"],
        session=session,
        tracking_server_arn=constants["tracking_server_arn"],
        train_instance_count_param=parameters["train_instance_count_param"],
        train_instance_type_param=parameters["train_instance_type_param"],
        processing_step=processor_step,
    )

    evaluator_step = get_evaluator_step(
        xgboost_container=xgboost_container,
        bucket_name=constants["bucket_name"],
        version=parameters["version"],
        project=constants["project"],
        feature_group_name=constants["feature_group_name"],
        tracking_server_arn=constants["tracking_server_arn"],
        train_instance_count_param=parameters["train_instance_count_param"],
        train_instance_type_param=parameters["train_instance_type_param"],
        region=constants["region"],
        processing_step=processor_step,
        training_step=trainer_step,
    )

    register_step = get_register_step(
        project=constants["project"],
        model_package_group_name=constants["model_package_group_name"],
        xgboost_container=xgboost_container,
        training_step=trainer_step,
        evaluation_step=evaluator_step,
        session=session,
    )

    conditional_step = get_conditional_step(
        project=constants["project"],
        bucket_name=constants["bucket_name"],
        version=parameters["version"],
        cumulative_return_threshold=parameters["cumulative_return_threshold"],
        accuracy_threshold=parameters["accuracy_threshold"],
        evaluation_step=evaluator_step,
        register_step=register_step,
    )

    return Pipeline(
        name=f"{constants['project']}-pipeline",
        parameters=[parameters[key] for key in parameters],
        pipeline_definition_config=pipeline_def_config,
        steps=[
            collector_step,
            processor_step,
            trainer_step,
            evaluator_step,
            conditional_step,
        ],
    )


if __name__ == "__main__":
    parameters = get_parameters()

    # print current directory
    print(os.getcwd())

    with open(os.path.join("src", "pipeline", "constants.json")) as f:
        constants = json.load(f)

    session = sagemaker.Session(boto3.Session(region_name=constants["region"]))

    sklearn_image_uri = image_uris.retrieve(
        framework="sklearn",
        region=constants["region"],
        version=constants["sklearn_image_uri_version"],
    )
    xgboost_container = image_uris.retrieve(
        framework="xgboost",
        region=constants["region"],
        version=constants["xgboost_container_version"],
    )

    pipeline = get_pipeline(
        session=session,
        parameters=parameters,
        constants=constants,
        sklearn_image_uri=sklearn_image_uri,
        xgboost_container=xgboost_container,
    )

    pipeline_definition_str = pipeline.definition()
    pipeline_definition_json = json.loads(pipeline_definition_str)
    # with open("pipeline_definition.json", "w") as file:
    #     json.dump(pipeline_definition_json, file, indent=4)

    pipeline.upsert(role_arn=sagemaker.get_execution_role())

from sagemaker.workflow.functions import Join
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor


def get_evaluator_step(
    xgboost_container: str,
    bucket_name: str,
    version,
    project: str,
    feature_group_name: str,
    tracking_server_arn: str,
    train_instance_count_param,
    train_instance_type_param,
    region: str,
    processing_step,
    training_step,
):
    evaluation_processor = ScriptProcessor(
        command=["python3"],
        image_uri=xgboost_container,
        role=sagemaker.get_execution_role(),
        instance_count=train_instance_count_param,
        instance_type=train_instance_type_param,
    )

    return ProcessingStep(
        name=f"{project}-evaluation",
        processor=evaluation_processor,
        code="src/models/evaluator.py",
        inputs=[
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "dataset_sizes"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=Join(
                    on="/",
                    values=[f"s3://{bucket_name}", "evaluation_results", version],
                ),
                output_name="evaluation_result",
            )
        ],
        job_arguments=[
            "--mode",
            "feature_store",
            "--input_path",
            "/opt/ml/processing/input",
            "--dataset_sizes_path",
            "/opt/ml/processing/input/dataset_sizes.json",
            "--data_version",
            version,
            "--target_column",
            "close_target",
            "--columns_to_drop",
            "write_time,api_invocation_time,is_deleted,datetime,type,version",
            "--model_path",
            "/opt/ml/processing/model",
            "--output_path",
            "/opt/ml/processing/output",
            "--feature_group_name",
            feature_group_name,
            "--region",
            region,
            "--bucket_name",
            bucket_name,
            "--tracking_server_arn",
            tracking_server_arn,
            "--experiment_name",
            f"{project}-evaluation-pipeline",
        ],
    )

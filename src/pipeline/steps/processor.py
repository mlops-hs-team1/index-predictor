from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.functions import Join
import sagemaker


def get_processor_step(
    project: str,
    version,
    bucket_name: str,
    process_instance_count_param,
    process_instance_type_param,
    sklearn_image_uri: str,
    collection_step,
    feature_group_name: str,
    region: str,
):
    processing_processor = ScriptProcessor(
        command=["python3"],
        image_uri=sklearn_image_uri,
        role=sagemaker.get_execution_role(),
        instance_count=process_instance_count_param,
        instance_type=process_instance_type_param,
    )

    return ProcessingStep(
        name=f"{project}-processing",
        processor=processing_processor,
        code="src/data/processor.py",
        inputs=[
            ProcessingInput(
                source=collection_step.properties.ProcessingOutputConfig.Outputs[
                    "raw_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=Join(
                    on="/", values=[f"s3://{bucket_name}", "data", "processed", version]
                ),
                output_name="dataset_sizes",
            )
        ],
        job_arguments=[
            "--mode",
            "feature_store",
            "--raw_data_filename",
            "/opt/ml/processing/input/data.csv",
            "--output_path",
            "/opt/ml/processing/output",
            "--version",
            version,
            "--feature_group_name",
            feature_group_name,
            "--region",
            region,
        ],
    )

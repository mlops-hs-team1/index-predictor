from sagemaker.processing import ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.functions import Join
import sagemaker


def get_collector_step(
    project: str,
    version,
    bucket_name: str,
    process_instance_count_param,
    process_instance_type_param,
    sklearn_image_uri: str,
):
    collection_processor = ScriptProcessor(
        command=["python3"],
        image_uri=sklearn_image_uri,
        role=sagemaker.get_execution_role(),
        instance_count=process_instance_count_param,
        instance_type=process_instance_type_param,
    )

    return ProcessingStep(
        name=f"{project}-collection",
        processor=collection_processor,
        code="../src/data/collector.py",
        inputs=[],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/data/raw",
                destination=Join(
                    on="/", values=[f"s3://{bucket_name}", "data", "raw", version]
                ),
                output_name="raw_data",
            )
        ],
        job_arguments=[
            "--mode",
            "train-val-test",
            "--version",
            version,
        ],
    )

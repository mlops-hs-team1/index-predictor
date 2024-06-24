from sagemaker.workflow.functions import Join
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


def get_trainer_step(
    xgboost_container: str,
    bucket_name: str,
    version,
    project: str,
    feature_group_name: str,
    session: sagemaker.Session,
    tracking_server_arn: str,
    train_instance_count_param,
    train_instance_type_param,
    processing_step,
):
    estimator = Estimator(
        image_uri=xgboost_container,
        role=sagemaker.get_execution_role(),
        instance_count=train_instance_count_param,
        instance_type=train_instance_type_param,
        output_path=Join(on="/", values=[f"s3://{bucket_name}", "models", version]),
        sagemaker_session=session,
        entry_point="../src/models/trainer.py",
    )

    estimator.set_hyperparameters(
        mode="feature_store",
        dataset_sizes_path="/opt/ml/input/data/dataset_sizes/dataset_sizes.json",
        data_version=version,
        target_column="close_target",
        columns_to_drop="write_time,api_invocation_time,is_deleted,datetime,type,version",
        model_output_path="/opt/ml/model",
        num_trials=10,
        feature_group_name=feature_group_name,
        bucket_name=bucket_name,
        region=session.boto_region_name,
        tracking_server_arn=tracking_server_arn,
        experiment_name=f"{project}-training-pipeline",
    )

    dataset_sizes_input = TrainingInput(
        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
            "dataset_sizes"
        ].S3Output.S3Uri
    )

    return TrainingStep(
        name=f"{project}-training",
        estimator=estimator,
        inputs={"dataset_sizes": dataset_sizes_input},
    )

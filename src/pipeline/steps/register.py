from sagemaker.model_metrics import MetricsSource
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics
import sagemaker


def get_register_step(
    project: str,
    model_package_group_name: str,
    xgboost_container: str,
    training_step,
    evaluation_step,
    session: sagemaker.Session,
):
    model = Model(
        image_uri=xgboost_container,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        name=f"{project}-model",
        sagemaker_session=session,
        role=sagemaker.get_execution_role(),
    )

    return RegisterModel(
        name=f"{project}-register",
        model=model,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs[
                    "evaluation_result"
                ].S3Output.S3Uri,
                content_type="application/json",
            )
        ),
        description="XGBoost model for index prediction, trained on pipeline",
    )

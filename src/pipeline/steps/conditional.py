from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet


def get_conditional_step(
    project: str,
    bucket_name: str,
    version,
    cumulative_return_threshold,
    accuracy_threshold,
    evaluation_step,
    register_step,
):
    step_fail = FailStep(
        name=f"{project}-fail",
        error_message=Join(
            on=" ",
            values=[
                "Execution failed due to Cumulative Return below",
                cumulative_return_threshold,
                "or Accuracy below",
                accuracy_threshold,
            ],
        ),
    )

    cond_cumulative_return = ConditionGreaterThan(
        left=JsonGet(
            step=evaluation_step.name,
            s3_uri=Join(
                on="/",
                values=[
                    f"s3://{bucket_name}",
                    "evaluation_results",
                    version,
                    "evaluation_report.json",
                ],
            ),
            json_path="cumulative_return",
        ),
        right=cumulative_return_threshold,
    )

    cond_accuracy = ConditionGreaterThan(
        left=JsonGet(
            step=evaluation_step.name,
            s3_uri=Join(
                on="/",
                values=[
                    f"s3://{bucket_name}",
                    "evaluation_results",
                    version,
                    "evaluation_report.json",
                ],
            ),
            json_path="test_accuracy",
        ),
        right=accuracy_threshold,
    )

    return ConditionStep(
        name=f"{project}-check",
        conditions=[cond_cumulative_return, cond_accuracy],
        if_steps=[register_step],
        else_steps=[step_fail],
        depends_on=[evaluation_step],
    )

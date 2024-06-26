import json
import boto3
from tqdm import trange
import numpy as np

s3 = boto3.client("s3")


def generate_endpoint_traffic(predictor, data):
    l = len(data)
    for i in trange(l):
        predictions = np.array(
            predictor.predict(data.iloc[i].values), dtype=float
        ).squeeze()


def get_file_list(bucket, prefix):
    try:
        files = [
            f.get("Key")
            for f in s3.list_objects(Bucket=bucket, Prefix=prefix).get("Contents")
        ]
        print(f"Found {len(files)} files in s3://{bucket}/{prefix}")

        return files
    except TypeError:
        print(f"No files found in s3://{bucket}/{prefix}")
        return []


def get_latest_data_capture_s3_url(bucket, prefix):
    capture_files = get_file_list(bucket, prefix)

    if capture_files:
        latest_data_capture_s3_url = (
            f"s3://{bucket}/{'/'.join(capture_files[-1].split('/')[:-1])}"
        )

        print(f"Latest data capture S3 url: {latest_data_capture_s3_url}")

        return latest_data_capture_s3_url
    else:
        return None


def get_latest_monitoring_report_s3_url(job_name):
    monitor_job = sm.list_processing_jobs(
        NameContains=job_name, SortOrder="Descending", MaxResults=2
    )["ProcessingJobSummaries"][0]["ProcessingJobName"]

    monitoring_job_output_s3_url = sm.describe_processing_job(
        ProcessingJobName=monitor_job
    )["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]

    print(f"Latest monitoring report S3 url: {monitoring_job_output_s3_url}")

    return monitoring_job_output_s3_url


def load_json_from_file(file_s3_url):
    bucket = file_s3_url.split("/")[2]
    key = "/".join(file_s3_url.split("/")[3:])
    print(f"Load JSON from: {bucket}/{key}")

    return json.loads(
        s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    )


def get_latest_monitor_execution(monitor):
    mon_executions = monitor.list_executions()

    if len(mon_executions):
        latest_execution = mon_executions[-1]  # get the latest execution
        latest_execution.wait(logs=False)

        print(
            f"Latest execution status: {latest_execution.describe().get('ProcessingJobStatus')}"
        )
        print(
            f"Latest execution result: {latest_execution.describe().get('ExitMessage')}"
        )

        latest_job = latest_execution.describe()
        if latest_job["ProcessingJobStatus"] != "Completed":
            print("No completed executions to inspect further")
        else:
            report_uri = latest_execution.output.destination
            print(f"Report Uri: {report_uri}")

        return latest_execution
    else:
        print("No monitoring schedule executions found")
        return None

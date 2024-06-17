import xgboost as xgb
import json
import numpy as np


def model_fn(model_dir):
    model = xgb.XGBClassifier()
    model.load_model(f"{model_dir}/xgboost_model.v0.0.1.json")
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data["data"])
    else:
        raise ValueError("This model only supports application/json input")


def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    return prediction.tolist()


def output_fn(prediction, content_type):
    if content_type == "application/json":
        response = {"predictions": prediction}
        return json.dumps(response)
    else:
        raise ValueError("This model only supports application/json output")

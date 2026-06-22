import dagshub
import mlflow
import json


mlflow.set_tracking_uri("https://dagshub.com/mrvivekkumar7171/uber-demand-prediction.mlflow")
dagshub.init(repo_owner='mrvivekkumar7171', repo_name='uber-demand-prediction', mlflow=True)


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

model_path = load_model_information("run_information.json")["model_uri"]
model = mlflow.sklearn.load_model(model_path)


def test_load_model_from_registry():
    assert model is not None, "Failed to load model from registry"
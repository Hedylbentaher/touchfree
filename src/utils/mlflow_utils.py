"""Configuration commune de MLflow et lecture de params.yaml."""

import os
import re

import yaml


def load_params(path="params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment):
    """Utilise le serveur DagsHub si les variables sont definies, sinon local.

    Variables attendues pour DagsHub :
      MLFLOW_TRACKING_URI=https://dagshub.com/<user>/touchfree.mlflow
      MLFLOW_TRACKING_USERNAME=<user>
      MLFLOW_TRACKING_PASSWORD=<jeton>
    """
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def clean_key(name):
    """MLflow refuse les parentheses : 'metrics/mAP50(B)' -> 'metrics/mAP50_B'."""
    name = name.strip().replace("(", "_").replace(")", "")
    return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", name)


def flatten(prefix, value, out=None):
    """{'augment': {'fliplr': 0.5}} -> {'augment.fliplr': 0.5} pour log_params."""
    out = {} if out is None else out
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value
    return out

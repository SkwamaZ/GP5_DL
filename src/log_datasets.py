import json

import mlflow

from src.utils import ROOT, load_config, setup_mlflow

# Кладём готовые сплиты этапа 3 в DagsHub как артефакты, чтобы любой ран
# воспроизводился только из логов, без поиска данных на диске вручную.
# Запускать один раз после того, как сплиты пересобраны: python -m src.log_datasets

DATASETS = {
    "experiment_tabular": ["tabular_train.parquet", "tabular_val.parquet", "tabular_test.parquet"],
    "experiment_text": [
        "text_train.parquet", "text_val.parquet", "text_test.parquet", "class_weights.json",
    ],
}


def log_one(experiment, files):
    cfg = load_config()
    proc = ROOT / cfg["paths"]["data_processed"]
    setup_mlflow(experiment)
    tags = {"mlflow.source.name": "src/log_datasets.py", "kind": "dataset"}
    with mlflow.start_run(run_name="dataset", tags=tags):
        manifest = {}
        for name in files:
            path = proc / name
            mlflow.log_artifact(path, artifact_path="data")
            manifest[name] = path.stat().st_size
        mlflow.log_params({"seed": cfg["random_seed"], "n_files": len(files)})
        mlflow.log_dict(manifest, "data/manifest.json")
        print(experiment, "| dataset run:", mlflow.active_run().info.run_id)


def main():
    cfg = load_config()
    for key, files in DATASETS.items():
        log_one(cfg["mlflow"][key], files)


if __name__ == "__main__":
    main()

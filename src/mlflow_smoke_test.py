import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(ROOT / ".env")

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("smoke-test")

    with mlflow.start_run(run_name="smoke-test") as run:
        mlflow.log_param("hello", "dagshub")
        mlflow.log_metric("answer", 42)
        print(f"Run logged: {run.info.run_id}")

    print(f"Tracking URI: {uri}")
    print("Открой эксперимент 'smoke-test' на DagsHub, чтобы увидеть run")


if __name__ == "__main__":
    main()

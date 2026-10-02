"""Pull models:/hotel-cancellation@champion into models/champion for the image build."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import mlflow


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is required")
    mlflow.set_tracking_uri(tracking_uri)
    download_root = Path("models/downloaded")
    if download_root.exists():
        shutil.rmtree(download_root)
    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri="models:/hotel-cancellation@champion",
            dst_path=str(download_root),
        )
    )
    destination = Path("models/champion")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(downloaded, destination)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias("hotel-cancellation", "champion")
    info_file = Path(
        mlflow.artifacts.download_artifacts(
            run_id=version.run_id,
            artifact_path="model_info.json",
        )
    )
    shutil.copy2(info_file, destination / "model_info.json")
    print(f"champion version {version.version} saved to {destination}")


if __name__ == "__main__":
    main()

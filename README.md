# ml-pipeline

A chest CT scan classifier (VGG16 transfer learning, two classes) wrapped in a staged training pipeline with DVC, a small Flask app for predictions, and a GitHub Actions workflow that builds a Docker image and pushes it to Amazon ECR. It is a worked example of structuring an ML project as config-driven stages and shipping it as a container.

## What it does not do

- It is not a medical tool. The last recorded evaluation (`scores.json`) is loss 32.85 and accuracy 0.43 after two training epochs, which is not a useful classifier.
- The trained model is not in the repo. `/predict` needs `model/model.h5`, and nothing in this repo creates it for you.
- No tests. The CI workflow's lint and unit-test steps only `echo`.
- MLflow logging to DagsHub is written but switched off (`log_into_mlflow()` is commented out in the evaluation stage).

## Quickstart

Not verified end to end: training needs a large dependency install (TensorFlow, PyTorch), a download from Google Drive and a long run. Checked on 2026-10-07: `pip install -e . --no-deps` and `import cnnClassifier` work, and `app.py` and `main.py` compile.

```bash
pip install -r requirements.txt
python main.py        # ingest data, build VGG16 base, train, evaluate
```

`main.py` runs four stages in order: download and unzip the dataset from the Google Drive link in `config/config.yaml`, build the VGG16 base model, train, and evaluate on a 30% validation split. Training output lands in `artifacts/training/model.h5`. To serve predictions, copy that file to `model/model.h5`, then:

```bash
python app.py         # Flask on 0.0.0.0:8080
```

- `GET /` serves a page to upload an image.
- `POST /predict` takes JSON `{"image": "<base64>"}` and returns `[{"image": "Normal"}]` or `[{"image": "Adenocarcinoma Cancer"}]`.
- `GET /train` runs `python main.py` inside the request.

The stages can also be run through DVC (`dvc repro`) using `dvc.yaml`.

## How it works

```
config/config.yaml + params.yaml
        |
 stage_01 data ingestion -> stage_02 prepare base model (VGG16)
        |                          |
        +-----> stage_03 training <+-----> stage_04 evaluation -> scores.json
```

- `src/cnnClassifier/` is an installable package (`setup.py`). `config/configuration.py` reads `config/config.yaml` and `params.yaml` and builds typed config objects (`entity/`); each stage in `pipeline/` calls a class in `components/`.
- `params.yaml` holds the hyperparameters: image size 224x224x3, batch size 16, 2 epochs, learning rate 0.01, ImageNet weights, 2 classes.
- `dvc.yaml` declares the same four stages with their dependencies, params and outputs; `scores.json` is the tracked metric. `.dvc/config` has no remote configured.
- `Dockerfile` uses `python:3.8-slim-buster`, installs the requirements and runs `app.py`.
- `.github/workflows/main.yaml` is manual-only (`workflow_dispatch`): placeholder CI, then build and push to ECR using repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY_NAME`, `AWS_ECR_LOGIN_URI`), then a deploy job on a self-hosted runner that pulls and runs the image. It fails without those secrets and a runner.
- `research/` holds the exploratory notebooks the stages were built from.

## Status

Built from July 2024 to April 2025 (first and last commit dates). Archived: no further changes planned.

## Known limits

- The two largest committed files (`model/model.h5`, 59 MB, and `research/Chest-CT-Scan-data.zip`, 49 MB) were removed from the tree in 2026-10 to keep the repo light. They remain in git history.
- `requirements.txt` is mostly unpinned apart from `tensorflow==2.13.0`, `mlflow==2.13.2` and a few others. It lists PyTorch, which the code does not import.
- The deploy job passes AWS keys into the container as environment variables. Prefer an instance role.
- `setup.py` carries the original package metadata (author handle and repo name) from an earlier repo name.
- The deploy workflow (`.github/workflows/main.yaml`) is manual-only (`workflow_dispatch`) and needs the AWS secrets listed above plus a self-hosted runner.

## License

MIT, see [LICENSE](LICENSE).

# MLOps Pipeline with Kubeflow, DVC, and GitHub Actions

**Student Name:** Muhammad Ibrahim Ansari
**Student ID:** 22I-0586

## Project Overview
This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting housing prices using the Boston Housing dataset. The pipeline is orchestrated using Kubeflow Pipelines and includes:
- **Data Versioning:** Handled by DVC (Data Version Control).
- **Pipeline Orchestration:** Kubeflow Pipelines (Extraction -> Preprocessing -> Training -> Evaluation).
- **Continuous Integration:** GitHub Actions automates testing and pipeline compilation.
- **Experiment Tracking:** MLflow is integrated into the training component.

## Project Structure
- `data/`: Contains the raw dataset and DVC tracking files.
- `src/`: Python source code for pipeline steps and components.
  - `pipeline_components.py`: Defines the Kubeflow components (Load, Preprocess, Train, Evaluate).
  - `pipeline.py`: Defines the pipeline structure and volume mounts.
- `components/`: Compiled YAML definitions for Kubeflow components.
- `.github/workflows/`: CI automation script for testing pipeline compilation.

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone [https://github.com/M-ibby04/mlops-kubeflow-assignment.git](https://github.com/M-ibby04/mlops-kubeflow-assignment.git)
   cd mlops-kubeflow-assignment


## Install Dependencies:
pip install -r requirements.txt

## DVC Setup: The data is tracked via DVC. To pull the latest data (if remote storage is configured):
dvc pull


## Pipeline Walkthrough
Compile the Pipeline: Run the following command to generate the pipeline.yaml file:
python src/pipeline.py


## Run on Kubeflow:
1. **Open the Kubeflow Pipelines Dashboard.**
2. **Click Upload Pipeline and select the generated pipeline.yaml.**
3. **Click Create Run to execute the pipeline.**

**Note:** Local execution of the pipeline requires a running Minikube cluster with Kubeflow Pipelines installed. If running in a restricted network environment, ensuring kubectl can pull the required Docker images is necessary for the UI to load.
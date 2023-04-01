from distutils.command.config import config
from statistics import mode
from clearml import Task, TaskTypes
from clearml import Dataset
from pathlib import Path

from config import AppConfig
from training import main_actions


def main():
    task:Task = Task.init(project_name="deepfake_detection_project",
                     task_name="training", task_type=TaskTypes.training)
    clearml_params = {
        "dataset_id":"acbaaa0e2b2242d28404059e89e54da1"
    }
    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.training_dataset_path = Path(dataset_path)
    main_actions(config=config)          


if __name__ == "__main__":
    main()
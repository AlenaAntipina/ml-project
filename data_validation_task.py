from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from config import AppConfig
from data_validation import main_actions


def main():
    task:Task = Task.init(project_name="deepfake_detection_dataset_project",
                     task_name="data_validation", task_type=TaskTypes.data_processing)
    clearml_params = {
        "dataset_id":"b680ef741d944903a0fba27220a98b83"
    }
    
    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)


if __name__ == "__main__":
    main()
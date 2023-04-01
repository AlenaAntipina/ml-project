from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from config import AppConfig
from data_preparation import main_actions


def main():
    task:Task = Task.init(project_name="deepfake_detection_dataset_project",
                     task_name="data_preparaion", task_type=TaskTypes.data_processing)
    
    clearml_params = {
        "dataset_id":"172511de3915468199f607067af728d1"
    }

    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)
    dataset = Dataset.create(dataset_project="deepfake-prepeared-dataset-project", dataset_name="deepfake-prepeared-dataset")
    dataset.add_files(config.dataset_output_path)
    task.set_parameter("output_dataset_id", dataset.id) 
    dataset.upload()
    dataset.finalize()


if __name__ == "__main__":
    main()
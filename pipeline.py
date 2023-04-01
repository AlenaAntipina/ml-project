from clearml import PipelineController

def main():
    pipe = PipelineController(
        name="Training pipeline", project="deepfake_detection_project", version="0.0.1"
    )
    pipe.set_default_execution_queue("default")
    pipe.add_step(
        name='preparation_data',
        base_task_project='deepfake_detection_project',
        base_task_name='data_preparation',
        parameter_override={
            'General/dataset_id': "172511de3915468199f607067af728d1"},
    )
    pipe.add_step(
        name='validation_data',
        parents=['preparation_data'],
        base_task_project='deepfake_detection_project',
        base_task_name='data_validation',
        parameter_override={
            'General/dataset_id': "acbaaa0e2b2242d28404059e89e54da1"},
    )
    pipe.add_step(
        name='training_step',
        parents=['validation_data'],
        base_task_project='deepfake_detection_project',
        base_task_name='training',
        parameter_override={
            'General/dataset_id': "${preparation_data.parameters.dataset_id}"},
    )
    pipe.start(queue="default")

if __name__ == "__main__":
    main()
import kfp
from kfp import dsl

#Loading the components we created in Task 2 from their YAML files
load_op = kfp.components.load_component_from_file('components/data_extraction.yaml')
preprocess_op = kfp.components.load_component_from_file('components/data_preprocessing.yaml')
train_op = kfp.components.load_component_from_file('components/model_training.yaml')
evaluate_op = kfp.components.load_component_from_file('components/model_evaluation.yaml')

@dsl.pipeline(
    name='Boston Housing MLOps Pipeline',
    description='A pipeline that trains a model on Boston housing data.'
)
def boston_pipeline(
    data_url: str = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
    n_estimators: int = 100,
    max_depth: int = 10
):
    vop = dsl.VolumeOp(
        name="create_volume",
        resource_name="data-volume", 
        size="1Gi", 
        modes=dsl.VOLUME_MODE_RWO
    )

    #Load Data
    load_task = load_op(data_url=data_url).add_pvolumes({"/data": vop.volume})
    
    #Preprocess Data
    preprocess_task = preprocess_op(
        input_csv=load_task.outputs['output_csv']
    ).add_pvolumes({"/data": vop.volume})
    
    #Train Model
    train_task = train_op(
        train_csv=preprocess_task.outputs['train_csv'],
        n_estimators=n_estimators,
        max_depth=max_depth
    ).add_pvolumes({"/data": vop.volume})
    
    #Evaluate Model
    evaluate_task = evaluate_op(
        test_csv=preprocess_task.outputs['test_csv'],
        model_pkl=train_task.outputs['model_pkl']
    ).add_pvolumes({"/data": vop.volume})

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(boston_pipeline, 'pipeline.yaml')
    print("SUCCESS: Pipeline compiled to 'pipeline.yaml'")
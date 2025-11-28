import kfp
from kfp.components import create_component_from_func

# Data Extraction Component
def data_extraction(data_url: str, output_csv: kfp.components.OutputPath(str)):
    """
    Fetches the dataset. In a real MLOps environment, this would use 
    'dvc get' to pull from version control. Here we simulate it by 
    loading the raw data from a URL or local path.
    """
    import pandas as pd
    
    print(f"Extraction: Loading data from {data_url}...")
    # Simulating DVC extraction by reading the file directly
    df = pd.read_csv(data_url)
    
    # Save the data to the output path so the next component can use it
    df.to_csv(output_csv, index=False)
    print(f"Extraction: Data saved to {output_csv}")

# Data Preprocessing Component
def data_preprocessing(input_csv: kfp.components.InputPath(str), 
                       train_csv: kfp.components.OutputPath(str),
                       test_csv: kfp.components.OutputPath(str)):
    """
    Cleans, scales, and splits the data into train and test sets.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print("Preprocessing: Reading data...")
    df = pd.read_csv(input_csv)
    
    # Cleaning: Drop missing values
    df = df.dropna()
    
    # Splitting: 80% Train, 20% Test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Output the split files
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    print(f"Preprocessing: Split completed. Train shape: {train.shape}, Test shape: {test.shape}")

# Model Training Component (With MLflow)
def model_training(train_csv: kfp.components.InputPath(str),
                   model_pkl: kfp.components.OutputPath(str),
                   n_estimators: int = 100,
                   max_depth: int = 10):
    
    import pandas as pd
    import mlflow
    import mlflow.sklearn
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    
    print("Training: Loading training data...")
    df = pd.read_csv(train_csv)
    
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    #MLflow Setup 
    mlflow.set_tracking_uri("file:///mlflow") 
    mlflow.set_experiment("Boston_Housing_Experiment")
    
    with mlflow.start_run():
        #Log Hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        #Initialize and Train Model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        
        #Log Model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Training: Model trained and logged to MLflow.")
        
        #Save Model Artifact for Kubeflow passing
        with open(model_pkl, 'wb') as f:
            pickle.dump(model, f)

# Model Evaluation Component
def model_evaluation(test_csv: kfp.components.InputPath(str),
                     model_pkl: kfp.components.InputPath(str),
                     metrics_json: kfp.components.OutputPath(str)):

    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("Evaluation: Loading test data and model...")
    df = pd.read_csv(test_csv)
    X_test = df.drop('MEDV', axis=1)
    y_test = df['MEDV']
    
    with open(model_pkl, 'rb') as f:
        model = pickle.load(f)
        
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Evaluation Results -> MSE: {mse}, R2: {r2}")
    
    #Save metrics in Kubeflow format
    metrics = {
        "metrics": [
            {"name": "Mean Squared Error", "numberValue": mse},
            {"name": "R2 Score", "numberValue": r2},
        ]
    }
    
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f)

#Main Execution: Compile Components to YAML
if __name__ == '__main__':
    #Compile Extraction
    create_component_from_func(
        data_extraction, 
        output_component_file='components/data_extraction.yaml', 
        base_image='python:3.9',
        packages_to_install=['pandas']
    )
    
    #Compile Preprocessing
    create_component_from_func(
        data_preprocessing, 
        output_component_file='components/data_preprocessing.yaml', 
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn']
    )
    
    #Compile Training
    create_component_from_func(
        model_training, 
        output_component_file='components/model_training.yaml', 
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn', 'mlflow']
    )
    
    #Compile Evaluation
    create_component_from_func(
        model_evaluation, 
        output_component_file='components/model_evaluation.yaml', 
        base_image='python:3.9',
        packages_to_install=['pandas', 'scikit-learn']
    )
    
    print("SUCCESS: All YAML components created in the 'components/' directory.")
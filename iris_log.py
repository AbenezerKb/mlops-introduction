import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://localhost:8000")


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from iris_pipeline import plot_model, plot_feature, plot_features, load_dataset


X, y = datasets.load_iris(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
    "intercept_scaling":1,
    "fit_intercept":True
}


lr = LogisticRegression(**params)
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)



mlflow.set_experiment("MLflow Quickstart")
iris_df = load_dataset()

plot_features(iris_df)
plot_model(lr, X_test, y_test)


experiment_name = "my_experiment"

mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
client = mlflow.tracking.MlflowClient()

run = client.create_run(experiment.experiment_id)
with mlflow.start_run(run_id = run.info.run_id):

    
    mlflow.log_params(params)

    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    mlflow.log_metric("accuracy", accuracy)

    
    signature = infer_signature(X_train, lr.predict(X_train))

    
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("features.png")


    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )
    mlflow.sklearn.log_model(lr, "iris", input_example=X_test)
    
    print("Model saved successfully!")

    


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])


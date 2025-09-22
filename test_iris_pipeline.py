import pandas as pd
import pytest
from sklearn.datasets import load_iris
from iris_pipeline import load_dataset, train, get_accuracy
from sklearn.linear_model import LogisticRegression


def test_load_dataset():
    df = load_dataset()    
    iris = load_iris()

    assert isinstance(df, pd.DataFrame)
	
    assert df.shape[0] == len(iris.data)

    expected_columns = iris.feature_names + ["species", "species_name"]
    assert list(df.columns) == expected_columns
	
    assert set(df["species"].unique()) == set(iris.target)			


def test_get_accuracy():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(df)
    accuracy = get_accuracy(model, X_test, y_test)

    assert isinstance(accuracy, float)	
	
    assert 1 == accuracy


def test_train():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(df)
    
    assert isinstance(model, LogisticRegression)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
	 
    total_rows = len(X_train) + len(X_test)
    assert total_rows == df.shape[0]
	

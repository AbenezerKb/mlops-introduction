import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df.apply(
        lambda x: str(iris.target_names[int(x["species"])]), axis=1
    )
    return df


def plot_feature(df, feature):
    # Plot a histogram of one of the features
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["species_name"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()


if __name__ == "__main__":
    iris_df = load_dataset()
   
    plot_feature(iris_df, "sepal length (cm)")
    plot_features(iris_df)
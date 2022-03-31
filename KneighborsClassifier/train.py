import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dvc.api
from IPython.display import display

import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    bacc_score = balanced_accuracy_score(y_pred, y_pred)
    return acc_score, bacc_score


# LOAD TRAIN DATA
# DIT WERKT NIET DOOR
path = "data/prepared/beer_profile_and_ratings.csv"
repo = "https://github.com/stijnhering/PreTraineeship"
version = "<GIT COMMIT>"
remote = "storage"

data_url = dvc.api.get_url(path=path, repo=repo)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # mlflow.set_experiment("/my-experiment")

    try:
        with dvc.api.open(path, repo=repo) as fd:
            data = pd.read_csv(fd, sep=",", index_col="Name")
            display(data.head())
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

    le = LabelEncoder()
    encoded = le.fit_transform(data[["Style"]].values.ravel())
    data[["Style"]] = encoded.reshape(-1, 1)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    #  SET Y VALUE, THIS CAN ALSO BE DONE WITH AN "sys.argv" function if necesary
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["Style"], axis=1)
    test_x = test.drop(["Style"], axis=1)
    train_y = train[["Style"]]
    test_y = test[["Style"]]

    C = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    kernel = str(sys.argv[2]) if len(sys.argv) > 2 else "rbf"
    degree = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    decision_function_shape = str(sys.argv[4]) if len(sys.argv) > 4 else "ovo"

    with mlflow.start_run(run_name="SupportVectorMachine") as run:

        params = {
            "decision_function_shape": decision_function_shape,
            "C": C,
            "kernel": kernel,
            "degree": degree,
            "verbose": 1,
        }

        svm = SVC(**params)
        # Train
        svm.fit(train_x, train_y)

        # Predict
        classified_styles = svm.predict(test_x)

        (acc_score, bacc_score) = eval_metrics(test_y, classified_styles)

        print(
            f"SupportVectorMachine model (decision_function_shape= {decision_function_shape}, C_regularization={C}, kernel={kernel} and degree:{degree}):"
        )
        print(f"  accuracy_score: {acc_score}")
        print(f"  balanced_accuracy_score: {bacc_score}")

        # Log data params
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("input_rows", data.shape[0])
        mlflow.log_param("input_cols", data.shape[1])

        # Log artifacts: columns usded for modeling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv("../data/features.csv", header=False, index=False)
        mlflow.log_artifact("../data/features.csv")

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv("../data/targets.csv", header=False, index=False)
        mlflow.log_artifact("../data/targets.csv")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy_score", acc_score)
        mlflow.log_metric("balanced_accuracy_score", bacc_score)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.log_model(sk_model=svm, artifact_path="sklearn-model", registered_model_name="sk-learn-gradBoost")
        else:
            mlflow.sklearn.log_model(svm, "model")

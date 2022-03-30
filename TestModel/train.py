import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dvc.api
from IPython.display import display

import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(y_true, y_pred):
    accuracy_score = accuracy_score(y_true, y_pred)
    balanced_accuracy_score = balanced_accuracy_score(y_pred, y_pred)
    top_k_accuracy_score = top_k_accuracy_score(y_pred, y_pred, k=5)
    average_precision_score = average_precision_score(y_pred, y_pred)
    roc_auc_score = roc_auc_score(y_pred, y_pred, multi_class='ovr')
    return accuracy_score, balanced_accuracy_score, top_k_accuracy_score, average_precision_score, roc_auc_score


# LOAD TRAIN DATA
path="data/prepared/beer_profile_and_ratings.csv"
repo="https://github.com/stijnhering/PreTraineeship"
version="<GIT COMMIT>"
remote="storage"

data_url = dvc.api.get_url(path=path, repo=repo)
mlflow.set_experiment("demo")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = pd.read_csv(data_url, sep=",")
        display(data)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    #  SET Y VALUE, THIS CAN ALSO BE DONE WITH AN "sys.argv" function if necesary
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["Style"], axis=1)
    test_x = test.drop(["Style"], axis=1)
    train_y = train[["Style"]]
    test_y = test[["Style"]]

    n_estimators = float(sys.argv[1]) if len(sys.argv) > 1 else 100
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    max_depth = float(sys.argv[3]) if len(sys.argv) > 3 else 5


    with mlflow.start_run(run_name="GradientBoosterRun") as run:

        params = {"n_estimators": 100,
                  "learning_rate":1.0,
                  "max_depth":5,}

        gradBoost = GradientBoostingClassifier(**params)

        gradBoost.fit(train_x, train_y)

        classified_styles = gradBoost.predict(test_x)

        (accuracy_score,
         balanced_accuracy_score,
         top_k_accuracy_score,
         average_precision_score,
         roc_auc_score) = eval_metrics(test_y, classified_styles)

        print(f"GradientBoostingClassifier model (n_estimators= {n_estimators}, learning_rate={learning_rate} and max_depth={max_depth}):")
        print(f"  accuracy_score: {accuracy_score}")
        print(f"  balanced_accuracy_score: {balanced_accuracy_score}")
        print(f"  top_k_accuracy_score: {top_k_accuracy_score}")
        print(f"  average_precision_score: {average_precision_score}")
        print(f"  roc_auc_score: {roc_auc_score}")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy_score", accuracy_score)
        mlflow.log_metric("balanced_accuracy_score", balanced_accuracy_score)
        mlflow.log_metric("top_k_accuracy_score", top_k_accuracy_score)
        mlflow.log_metric("average_precision_score", average_precision_score)
        mlflow.log_metric("roc_auc_score", roc_auc_score)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.log_model(sk_model=gradBoost,
                            artifact_path="sklearn-model",
                            registered_model_name="sk-learn-gradBoost")
        else:
            mlflow.sklearn.log_model(gradBoost, "model")
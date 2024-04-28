import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from model_selection import split_data

# Store each classifier and its parameter grid.
classifiers = {    
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 10], "weights": ["uniform", "distance"]}),
    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [5, 10], "min_samples_split": [2, 5]}),
    "Logistic Regression": (LogisticRegression(tol=1e-3, max_iter=1000), {"solver": ["sag", "saga"], "C": [1.0, 10.0]}),
    "Neural Network": (MLPClassifier(max_iter=200, random_state=42, tol=0.001), {'hidden_layer_sizes': [(10,), (50,)], 'activation': ['tanh', 'relu'],})

}

def find_best_hyperparameters(classifier, param_grid, x_train, y_train, cv=3):
    """
    Find the optimal parameters for a given classifier using RandomizedSearchCV.

    Parameters:
    - classifier: sklearn classifier object, the classifier to tune
    - param_grid: dict, the parameter grid to search
    - x_train: array-like of shape (# samples, # features), the training input samples
    - y_train: array-like of (# samples,), the class labels
    - cv: int, optional, the number of folds for cross-validation (default: 5)

    Returns:
    - best_params: dict, the optimal parameters found by RandomizedSearchCV
    """
    rand_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, n_iter=4)
    rand_search.fit(x_train, y_train)
    best_params = rand_search.best_params_
    return best_params

def tune_hyperparameters(dataset, train_df, test_df):
    """
    Tune each classifier in classifiers, finding the best hyperparameters for using RandomizedSearchCV,
    and organize the results into a JSON object.

    Parameters:
    - dataset: str, either "kickstarter" or "indiegogo"
    - train_df: pandas DataFrame, the feature selected training dataset used to tune the hyperparameters
    - test_df: pandas DataFrame, the feature selected testing dataset used to assess the tuned models

    Returns:
    - results: dict, a JSON object containing the best parameters and performance for each classifier
    """
    results = {}
    for name, (classifier, param_grid) in classifiers.items():
        print(f'{name}, {dataset}')
        # Split the data
        if dataset == "kickstarter":
            x_train = train_df.drop(columns=["State"]) 
            y_train = train_df["State"]
            x_test = test_df.drop(columns=["State"])
            y_test = test_df["State"]
        else:
            x_train = train_df.drop(columns=["state"]) 
            y_train = train_df["state"]
            x_test = test_df.drop(columns=["state"])
            y_test = test_df["state"]

        # Find the best hyperparameters
        best_params = find_best_hyperparameters(classifier, param_grid, x_train, y_train)

        # Train the classifier with the best hyperparameters
        classifier.set_params(**best_params)
        classifier.fit(x_train, y_train)

        # Evaluate the classifier
        accuracy = classifier.score(x_test, y_test)

        # Store results
        results[name] = {
            "best_parameters": best_params,
            "accuracy": accuracy
        }

    # Convert dictionary to JSON and save to file
    with open(f'{dataset}_best_parameters.json', "w") as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    # Tune parameters using the kickstarter dataset
    kickstarter_train = pd.read_csv("../data/selected/kickstarter_train_selected.csv")
    kickstarter_test = pd.read_csv("../data/selected/kickstarter_test_selected.csv")
    tune_hyperparameters("kickstarter", kickstarter_train, kickstarter_test)

    # Tune parameters using the indiegogo dataset
    indiegogo_train = pd.read_csv("../data/selected/indiegogo_train_selected.csv")
    indiegogo_test = pd.read_csv("../data/selected/indiegogo_test_selected.csv")
    tune_hyperparameters("indiegogo", indiegogo_train, indiegogo_test)

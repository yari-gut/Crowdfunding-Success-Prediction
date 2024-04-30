import json
import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

def evaluate_model(clf, model, x_train, y_train, x_test, y_test):
    """
    Given a tuned classifier, train and evaluate the model.

    Parameters:
    - clf : sklearn.ClassifierMixin, The sklearn classifier model 
    - model: str, name of the classifier

    Returns:
    - result_dict: dict with the keys "AUC", "AUPRC", "F1", "Time". and the values floats.
    """
    # Start time
    start = time.time()

    # Train the classifier
    clf.fit(x_train, y_train)

    timeElapsed = time.time() - start

    # Evaluate on test set
    yHat = clf.predict(x_test)
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(x_test)[:, 1])
    auc_score = auc(fpr, tpr)
    auprc = auc(recall, precision)
    f1 = f1_score(y_test, yHat)

    # Create result/roc dictionaries
    result_dict = {"model": model, "AUC": auc_score, "AUPRC": auprc, "F1": f1, "Time": timeElapsed}
    roc = {"model": model, "fpr": fpr, "tpr": tpr}

    return result_dict, roc

def eval_models(dataset):
    """
    Evaluate each tuned model for the given dataset.

    Parameters:
    - dataset: str, either "kickstarter" or "indiegogo"

    Returns:
    - result_df: pandas DataFrame, with columns "model", "AUC", "AUPRC", "F1", and "Time"
    - roc_df: pandas DataFrame, with columns "model", "fpr", and "tpr"
    """
    # Load parameters from JSON file, and x_train, x_test, y_train, and y_test from CSVs.
    if dataset == "indiegogo":
        with open("indiegogo_best_parameters.json", "r") as file:
            params = json.load(file)
        train = pd.read_csv("../data/selected/indiegogo_train_selected.csv")
        x_train = train.drop(columns=["state"]) 
        y_train = train["state"]
        test = pd.read_csv("../data/selected/indiegogo_test_selected.csv")
        x_test = test.drop(columns=["state"]) 
        y_test = test["state"]
    elif dataset == "kickstarter":
        with open("kickstarter_best_parameters.json", "r") as file:
            params = json.load(file)
        train = pd.read_csv("../data/selected/kickstarter_train_selected.csv")
        x_train = train.drop(columns=["State"]) 
        y_train = train["State"]
        test = pd.read_csv("../data/selected/kickstarter_test_selected.csv")
        x_test = test.drop(columns=["State"]) 
        y_test = test["State"]
    else:
        raise ValueError("Invalid value for 'dataset'. Value must be 'kickstarter' or 'indiegogo'.")

    # Store evaluation results
    result = []
    roc_df = pd.DataFrame()
    
    # Loop through each tuned model and evaluate it
    for model in params:
        if model == "Random Forest":
            clf = RandomForestClassifier(**params[model])
        elif model == "KNN":
            clf = KNeighborsClassifier(**params[model])
        elif model == "Decision Tree":
            clf = DecisionTreeClassifier(**params[model])
        elif model == "Logistic Regression":
            clf = LogisticRegression(**params[model], tol=1e-3, max_iter=1000)
        elif model == "Neural Network":
            clf = MLPClassifier(**params[model], max_iter=300, random_state=42, tol=0.001)
        else:
            raise KeyError(f"Invalid key. The model '{model}' has not been tuned.")
        result_dict, roc_dict = evaluate_model(clf, model, x_train, y_train, x_test, y_test)
        roc_df = pd.concat([roc_df, pd.DataFrame(roc_dict)], ignore_index=True)
        result.append(result_dict)

    result_df = pd.DataFrame(result)
    return result_df, roc_df

def plot_roc(dataset, roc_df):
    """
    Evaluate each tuned model for the given dataset.

    Parameters:
    - dataset: str, either "kickstarter" or "indiegogo"
    - roc_df: pandas DataFrame, with columns "model", "fpr", and "tpr"

    Returns: None
    """
    models = roc_df['model'].unique()

    for model in models:
        subset = roc_df[roc_df['model'] == model]
        plt.plot(subset['fpr'], subset['tpr'], label=model)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve by Model ({dataset})')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # Evaluate Indiegogo
    indiegogo_results, indiegogo_roc = eval_models("indiegogo")
    print(indiegogo_results)
    plot_roc("indiegogo", indiegogo_roc)

    # Evaluate Kickstarter
    kickstarter_results, kickstarter_roc = eval_models("kickstarter")
    print(kickstarter_results)
    plot_roc("kickstarter", kickstarter_roc)

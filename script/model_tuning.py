import pandas as pd
import json
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(df):
    """
    Split a pandas DataFrame into training and testing sets (80-20 split)
    and save them as CSV files.
    
    Parameters:
    - df: pandas DataFrame, the DataFrame to split

    Returns:
    - train_df: pandas DataFrame, the training set
    - test_df: pandas DataFrame, the testing set
    """
    # Conduct an 80-20 train-test split
    train_df, test_df = train_test_split(df, test_size=0.2)
    return train_df, test_df

def find_best_hyperparameters(classifier, param_grid, x_train, y_train, cv=5):
    """
    Find the optimal parameters for a given classifier using Grid Search CV.

    Parameters:
    - classifier: sklearn classifier object, the classifier to tune
    - param_grid: dict, the parameter grid to search
    - x_train: array-like of shape (# samples, # features), the training input samples
    - y_train: array-like of (# samples,), the class labels
    - cv: int, optional, the number of folds for cross-validation (default: 5)

    Returns:
    - best_params: dict, the optimal parameters found by Grid Search CV
    """
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv)
    grid_search.fit(x_train, y_train)
    
    best_params = grid_search.best_params_
    
    return best_params

# Store each classifier and its parameter grid.
classifiers = {
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}),
    'Logistic Regression': (LogisticRegression(), {'C': [0.1, 1.0, 10.0], 'solver': ['sag', 'saga'],}),
    'Naive Bayes': (MultinomialNB(), {'alpha': [0.1, 0.5, 1.0, 2.0]})
}

def tune_hyperparameters(df):
    """
    Tune each classifier in classifiers, finding the best hyperparameters for using Grid Search CV,
    and organize the results into a JSON object.

    Parameters:
    - df: pandas DataFrame, the DataFrame used to tune the hyperparameters

    Returns:
    - results: dict, a JSON object containing the best parameters and performance for each classifier
    """
    results = {}
    for name, (classifier, param_grid) in classifiers.items():
        print(name)
        # Split the data
        train_df, test_df = split_data(df)
        x_train = train_df.drop(columns=['State']) 
        y_train = train_df['State']
        x_test = test_df.drop(columns=['State'])
        y_test = test_df['State']

        # Find the best hyperparameters
        best_params = find_best_hyperparameters(classifier, param_grid, x_train, y_train)

        # Train the classifier with the best hyperparameters
        classifier.set_params(**best_params)
        classifier.fit(x_train, y_train)

        # Evaluate the classifier
        accuracy = classifier.score(x_test, y_test)

        # Store results
        results[name] = {
            'best_parameters': best_params,
            'accuracy': accuracy
        }

    # Convert dictionary to JSON and save to file
    with open('best_parameters.json', 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    # Load the kickstarter dataset
    kickstarter_data = pd.read_csv("../data/kickstarter_projects.csv")

    # Preprocess data (temp)
    text_columns= ['ID', 'Name', 'Category', 'Subcategory', 'Country', 'Launched', 'Deadline']
    kickstarter_data = kickstarter_data.drop(text_columns, axis=1)
    encoder = LabelEncoder()
    kickstarter_data['State'] = encoder.fit_transform(kickstarter_data['State'])
    scaler = MinMaxScaler()
    kickstarter_data[['Goal', 'Pledged', 'Backers']] = scaler.fit_transform(kickstarter_data[['Goal', 'Pledged', 'Backers']])

    tune_hyperparameters(kickstarter_data)

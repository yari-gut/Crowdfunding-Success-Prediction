import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def split_data(df):
    """
    Split a pandas DataFrame into a train and test set (80-20 split).
    
    Parameters:
    - df: pandas DataFrame, the DataFrame to split

    Returns:
    - train_df: pandas DataFrame, the training set
    - test_df: pandas DataFrame, the testing set
    """
    # Conduct an 80-20 train-test split
    train_df, test_df = train_test_split(df, test_size=0.2)
    return train_df, test_df

def cal_corr(df):
    """
    Compute the Pearson correlation matrix
    
    Parameters
    - df: pandas.DataFrame, the dataset

    Returns
    - corrDF: pandas.DataFrame, the correlation between the different columns
    """
    # Get the Pearson correlation matrix
    corrDF = df.corr(method="pearson")

    return corrDF

def select_features(trainDF, testDF, dataset):
    """
    Preprocess the features
    
    Parameters
    - trainDF: pandas.DataFrame, the training dataframe
    - testDF: pandas.DataFrame, the test dataframe

    Returns
    - trainDF: pandas.DataFrame, return the feature-selected trainDF dataframe
    - testDF: pandas.DataFrame, return the feature-selected testDT dataframe
    """
    corr_matrix = cal_corr(trainDF)

    # Set delta and gamma thresholds based on the dataset
    if dataset == "kickstarter":
        delta = 0.8
        gamma = 0.1
    elif dataset == "indiegogo":
        delta = 0.85  
        gamma = 0.15  
    else:
        raise ValueError("Invalid dataset name. Please specify either 'kickstarter' or 'indiegogo'.")

    # Find features highly correlated with each other
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j] > delta):
                corr_i_target = abs(corr_matrix.iloc[i, -1])
                corr_j_target = abs(corr_matrix.iloc[j, -1])

                # Remove the feature with lower correlation to the target feature
                if corr_i_target < corr_j_target:
                    correlated_features.add(corr_matrix.columns[i])
                else:
                    correlated_features.add(corr_matrix.columns[j])

    # Find features with low correlation to the target variable
    low_correlation_features = set()
    if (dataset == "kickstarter"):
        target = trainDF.columns[trainDF.columns.get_loc("State")]
    else:
        target = trainDF.columns[trainDF.columns.get_loc("state")]
    target_corr = corr_matrix[target]
    low_correlation_features.update(target_corr[abs(target_corr) < gamma].index)

    remove = correlated_features.union(low_correlation_features)

    # Remove features from both training and test datasets
    trainDF = trainDF.drop(columns=remove)
    testDF = testDF.drop(columns=remove)

    return trainDF, testDF


def heatmap(train_df):
    # Calculate correlation matrices using cal_corr function
    corr_matrix_binary = cal_corr(train_df)

    # Plot correlation matrix for binary representation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_binary, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()

if __name__ == "__main__":
    kickstarter = pd.read_csv("../data/preprocessed/kickstarter_preprocessed.csv")
    kickstarter_train_df, kickstarter_test_df = split_data(kickstarter)
    print("Original Training Shape:", kickstarter_train_df.shape)
    heatmap(kickstarter_train_df)
    kickstarter_train_df, kickstarter_test_df = select_features(kickstarter_train_df, kickstarter_test_df, "kickstarter")
    print("Transformed Training Shape:", kickstarter_train_df.shape)
    # If a folder to store feature selected data does not exist, create it
    path = "../data/selected"
    if not os.path.exists(path):
        os.makedirs(path)
    # Save to csv
    kickstarter_train_df.to_csv('../data/selected/kickstarter_train_selected.csv', index=False)
    kickstarter_test_df.to_csv('../data/selected/kickstarter_test_selected.csv', index=False)

    indiegogo = pd.read_csv("../data/preprocessed/indiegogo_preprocessed.csv")
    indiegogo_train_df, indiegogo_test_df = split_data(indiegogo)
    print("Original Training Shape:", indiegogo_train_df.shape)
    heatmap(indiegogo_train_df)
    indiegogo_train_df, indiegogo_test_df = select_features(indiegogo_train_df, indiegogo_test_df, "indiegogo")
    print("Transformed Training Shape:", indiegogo_train_df.shape)
    # Save to csv
    indiegogo_train_df.to_csv('../data/selected/indiegogo_train_selected.csv', index=False)
    indiegogo_test_df.to_csv('../data/selected/indiegogo_test_selected.csv', index=False)

import os
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def preprocess_kickstarter(kickstarter_df):
    """
    Preprocess the Kickstarter dataset.
    
    Parameters:
    - kickstarter_df (DataFrame): DataFrame containing the raw Kickstarter dataset.
    
    Returns:
    - cleaned_df (DataFrame): DataFrame containing the preprocessed Kickstarter dataset.
    """
    # Drop samples with missing values for any feature
    cleaned_df = kickstarter_df.dropna()

    # Drop features that are not meaningful
    cleaned_df = cleaned_df.drop(labels=["ID"], axis=1)

    # Convert features related to date and time to POSIX timestamps
    datetime_features = ["Launched", "Deadline"]
    for feature in datetime_features:
        cleaned_df[feature] = pd.to_datetime(cleaned_df[feature]).astype('int64') // 10**9
    
    # Convert "Name" feature to numerical using feature hashing
    cleaned_df["Name"] = cleaned_df["Name"].str.split()
    hasher = FeatureHasher(n_features=6, input_type="string")
    hashed_features = hasher.transform(cleaned_df["Name"])
    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'Name_{i}' for i in range(6)])
    cleaned_df = pd.concat([cleaned_df, hashed_df], axis=1)
    cleaned_df = cleaned_df.drop(labels=["Name"], axis=1)

    # Convert categorical columns to numerical using label encoding
    categorical_features = ["Category", "Subcategory", "Country"]
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        cleaned_df[feature] = label_encoder.fit_transform(cleaned_df[feature])

    # Normalize numeric features
    numeric_features = ["Launched", "Deadline", "Goal", "Pledged", "Backers"]
    scaler = MinMaxScaler()
    cleaned_df[numeric_features] = scaler.fit_transform(cleaned_df[numeric_features])

    # Convert the target column "State" to binary, dropping samples that are not "Successful" or "Failed"
    cleaned_df = cleaned_df[cleaned_df["State"].isin(['Successful', 'Failed'])]
    cleaned_df["State"] = cleaned_df["State"].map({"Successful": 1, "Failed": 0})

    return cleaned_df

def preprocess_indiegogo(indiegogo_df):
    """
    Preprocess the Indiegogo dataset.
    
    Parameters:
    - indiegogo_df (DataFrame): DataFrame containing the raw Indiegogo dataset.
    
    Returns:
    - cleaned_df (DataFrame): DataFrame containing the preprocessed Indiegogo dataset.
    """
    # Drop samples with missing values for any feature
    cleaned_df = indiegogo_df.dropna()

    # Drop features that are not meaningful or redundant
    cleaned_df = cleaned_df.drop(labels=["project_id", "url", "amount_raised", "funded_percent", "category", "tperiod"], axis=1)

    # Create columns "launched" and "deadline" to condense columns related to date and time
    cleaned_df["launched"] = pd.to_datetime(cleaned_df['date_launch'] + " " + cleaned_df['time_launch']).astype('int64') // 10**9
    cleaned_df["end"] = pd.to_datetime(cleaned_df['date_end'] + " " + cleaned_df['time_end']).astype('int64') // 10**9
    condensed_features = ["year_launch", "month_launch", "day_launch", "time_launch", "year_end", "month_end", "day_end", "time_end", "date_launch", "date_end",
                          "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    cleaned_df = cleaned_df.drop(labels=condensed_features, axis=1)

    # Convert features related to descriptive text to numerical using feature hashing
    text_features = ["title", "tagline"]
    hashers = {feature: FeatureHasher(n_features=6, input_type='string') for feature in text_features}
    hashed_dfs = []
    for feature in text_features:
        cleaned_df[feature] = cleaned_df[feature].str.split()
        hashed_features = hashers[feature].fit_transform(cleaned_df[feature])
        hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'{feature}_{i}' for i in range(6)])
        hashed_dfs.append(hashed_df)
        cleaned_df.reset_index(drop=True, inplace=True)
        hashed_df.reset_index(drop=True, inplace=True)
        cleaned_df = pd.concat([cleaned_df, hashed_df], axis=1)
    cleaned_df = cleaned_df.drop(labels=text_features, axis=1)

    # Convert categorical features to numerical using one-hot encoding
    cleaned_df = pd.get_dummies(cleaned_df, columns=["currency"])

    # Normalize numeric features
    numeric_features = ["launched", "end", "amount_raised_usd", "goal_usd"]
    scaler = MinMaxScaler()
    cleaned_df[numeric_features] = scaler.fit_transform(cleaned_df[numeric_features])

    return cleaned_df

if __name__ == '__main__':
    # If a folder to store preprocessed data does not exist, create it
    path = "../data/preprocessed"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Preprocess the kickstarter dataset and save it as a CSV file
    kickstarter = pd.read_csv("../data/raw/kickstarter.csv")
    kickstarter_preprocessed = preprocess_kickstarter(kickstarter)
    kickstarter_preprocessed.to_csv('../data/preprocessed/kickstarter_preprocessed.csv', index=False)

    # Preprocess the indiegogo dataset and save it as a CSV file
    indiegogo = pd.read_csv("../data/raw/indiegogo.csv")
    indiegogo_preprocessed = preprocess_indiegogo(indiegogo)
    indiegogo_preprocessed.to_csv('../data/preprocessed/indiegogo_preprocessed.csv', index=False)

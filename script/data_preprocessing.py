import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_kickstarter_data(kickstarter_data):
    """
    Preprocess the Kickstarter dataset.
    
    Parameters:
    - kickstarter_data (DataFrame): DataFrame containing the Kickstarter dataset.
    
    Returns:
    - DataFrame: Preprocessed Kickstarter dataset.
    """
    # Drop unnecessary columns
    kickstarter_data.drop(['ID', 'Subcategory', 'Country'], axis=1, inplace=True)
    
    # Convert 'Launched' and 'Deadline' columns to datetime
    kickstarter_data['Launched'] = pd.to_datetime(kickstarter_data['Launched'])
    kickstarter_data['Deadline'] = pd.to_datetime(kickstarter_data['Deadline'])
    
    # Extract relevant information from datetime columns
    kickstarter_data['Campaign_Duration'] = (kickstarter_data['Deadline'] - kickstarter_data['Launched']).dt.days
    
    # Convert 'State' column to binary: Successful (1) or Failed (0)
    kickstarter_data['State'] = (kickstarter_data['State'] == 'successful').astype(int)
    
    # Normalize numerical features: 'Goal', 'Pledged', 'Backers'
    scaler = MinMaxScaler()
    kickstarter_data[['Goal', 'Pledged', 'Backers']] = scaler.fit_transform(kickstarter_data[['Goal', 'Pledged', 'Backers']])
    
    # Drop rows with missing values (if necessary)
    kickstarter_data.dropna(inplace=True)
    
    return kickstarter_data

def preprocess_indiegogo_data(indiegogo_data):
    """
    Preprocess the Indiegogo dataset.
    
    Parameters:
    - indiegogo_data (DataFrame): DataFrame containing the Indiegogo dataset.
    
    Returns:
    - DataFrame: Preprocessed Indiegogo dataset.
    """
    # Drop unnecessary columns
    indiegogo_data.drop(['currency', 'year_end', 'month_end', 'day_end', 'time_end', 'project_id', 'tagline', 'url', 'date_launch', 'date_end'], axis=1, inplace=True)
    
    # Convert 'year_launch', 'month_launch', 'day_launch', 'time_launch' columns to datetime
    indiegogo_data['year_launch'] = pd.to_datetime(indiegogo_data['year_launch'] + '-' + indiegogo_data['month_launch'] + '-' + indiegogo_data['day_launch'] + ' ' + indiegogo_data['time_launch'])
    
    # Drop 'month_launch', 'day_launch', 'time_launch' columns after converting to datetime
    indiegogo_data.drop(['month_launch', 'day_launch', 'time_launch'], axis=1, inplace=True)
    
    # Normalize numerical features: 'amount_raised_usd', 'goal_usd'
    scaler = MinMaxScaler()
    indiegogo_data[['amount_raised_usd', 'goal_usd']] = scaler.fit_transform(indiegogo_data[['amount_raised_usd', 'goal_usd']])
    
    # Drop rows with missing values (if necessary)
    indiegogo_data.dropna(inplace=True)
    
    return indiegogo_data

def align_column_names(indiegogo_data, kickstarter_data):
    """
    Align column names between Indiegogo and Kickstarter datasets.

    Parameters:
    - indiegogo_data (DataFrame): DataFrame containing the Indiegogo dataset.
    - kickstarter_data (DataFrame): DataFrame containing the Kickstarter dataset.

    Returns:
    - DataFrame: Indiegogo dataset with aligned column names.
    - DataFrame: Kickstarter dataset with aligned column names.
    """
    # Dictionary to map Indiegogo column names to Kickstarter column names
    indiegogo_to_kickstarter_mapping = {
        'category': 'Category',
        'amount_raised_usd': 'Pledged',
        'goal_usd': 'Goal'
        # Add more mappings as needed
    }

    # Rename columns in Indiegogo dataset
    indiegogo_data.rename(columns=indiegogo_to_kickstarter_mapping, inplace=True)

    # Dictionary to map Kickstarter column names to Indiegogo column names
    kickstarter_to_indiegogo_mapping = {
        'Name': 'title',
        'Goal': 'goal_usd',
        'Pledged': 'amount_raised_usd'
        # Add more mappings as needed
    }

    # Rename columns in Kickstarter dataset
    kickstarter_data.rename(columns=kickstarter_to_indiegogo_mapping, inplace=True)

    return indiegogo_data, kickstarter_data


def merge_datasets(kickstarter, indiegogo):
    """
    Merge the Kickstarter and Indiegogo datasets.
    
    Parameters:
    - kickstarter (DataFrame): DataFrame containing the Kickstarter dataset.
    - indiegogo (DataFrame): DataFrame containing the Indiegogo dataset.
    
    Returns:
    - DataFrame: Merged dataset.
    """
    # Align column names between the datasets
    align_column_names(indiegogo, kickstarter)

    # Concatenate the datasets
    merged_data = pd.concat([kickstarter, indiegogo], ignore_index=True)
    return merged_data

if __name__ == '__main__':
    # Load the two datasets
    kickstarter_data = pd.read_csv("kickstarter_projects.csv")
    indiegogo_data = pd.read_csv("indiegogo.csv")

    # Preprocess Kickstarter data
    kickstarter_data = preprocess_kickstarter_data(kickstarter_data)

    # Preprocess Indiegogo data
    indiegogo_data = preprocess_indiegogo_data(indiegogo_data)

    # Merge the preprocessed datasets
    merged_data = merge_datasets(kickstarter_data, indiegogo_data)

    # Save the merged dataset to a CSV file
    merged_data.to_csv('merged_dataset.csv', index=False)

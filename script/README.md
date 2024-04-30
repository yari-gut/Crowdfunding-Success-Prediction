# Data Preprocessing

The `data_preprocessing.py` script performs preprocessing tasks on Kickstarter and Indiegogo datasets before merging them. The preprocessing steps include:

- **Kickstarter Data Preprocessing:**

  - Drop unnecessary columns such as 'ID'.
  - Convert 'Launched' and 'Deadline' columns to POSIX timestamps.
  - Convert the 'State' column to binary, representing Successful (1) or Failed (0) campaigns.
  - Convert categorical columns ('Category', 'Subcategory', 'Country') to numeric using label encoding.
  - Convert 'Name' to numerical using feature hashing.
  - Normalize numerical features ('Launched', 'Deadline', 'Goal', 'Pledged', 'Backers') using Min-Max scaling.
  - Drop rows with missing values.

- **Indiegogo Data Preprocessing:**

  - Drop unnecessary or redundant columns such as 'project_id', 'url', 'amount_raised', 'funded_percent', 'category', and 'tperiod'.
  - Condense time-related features ('year_launch', 'month_launch', 'day_launch', 'time_launch', 'year_end', 'month_end', 'day_end', 'time_end', 'date_launch', 'date_end', 'mar', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec') to "launched" and "end", which are both POSIX timestamps.
  - Convert features consisting of long strings of text ('title', 'tagline') to numerical using feature hashing.
  - Convert 'currency" to numerical using one-hot encoding.
  - Normalize numerical features ('launched', 'end', 'amount_raised_usd', 'goal_usd') using Min-Max scaling.
  - Drop rows with missing values.

The script ensures that both datasets are properly preprocessed, facilitating further analysis and model training in subsequent stages of the project.

## Feature Selection and Data Splitting

The `model_selection.py` script handles feature selection and data splitting tasks. It includes the following functions:

- **split_data:** This function splits a pandas DataFrame into training and testing sets using an 80-20 split ratio.

- **cal_corr:** This function computes the Pearson correlation matrix for the dataset.

- **select_features:** Preprocesses the features by removing highly correlated features and features with low correlation to the target variable.

- **heatmap:** This function plots the correlation matrix as a heatmap using seaborn library.

The script also includes a main block that demonstrates the usage of these functions on both the Kickstarter and Indiegogo datasets. It loads the preprocessed datasets, splits them into training and testing sets, performs feature selection, visualizes the correlation matrix, and saves the feature-selected datasets to CSV files.

## Model Tuning

The `model_tuning.py` script tunes hyperparameters for various machine learning classifiers using Grid Search CV. It performs the following tasks:

- **Classifier Selection:** It defines a dictionary of classifiers, each paired with a parameter grid for hyperparameter tuning. The supported classifiers include:

  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
  - Naive Bayes

- **Hyperparameter Tuning:** For each classifier, the script utilizes Random Search CV to find the optimal combination of hyperparameters. It searches through the specified parameter grid and evaluates each combination using 3-fold cross-validation.

- **Best Parameters:** After hyperparameter tuning, the script identifies the best parameters for each classifier based on the highest roc_auc.

- **Results Storage:** The script organizes the results, including the best parameters and accuracy for each classifier, into a JSON object.

The script enables efficient optimization of machine learning classifiers by systematically searching for the best hyperparameters. It facilitates the selection of optimal models for predicting crowdfunding campaign success, enhancing the overall effectiveness of the project's predictive analytics.

# Model Evaluation

The `model_evaluation.py` script evaluates tuned machine learning models on the Kickstarter and Indiegogo crowdfunding datasets. It includes the following functions:

- **evaluate_model:** This function trains and evaluates a tuned classifier on the test set.

- **eval_models:** This function iterates through each tuned model, evaluates it, and organizes all results into pandas DataFrames.

- **plot_roc:** This function plots the ROC curve for each model.

The script includes a main block that demonstrates the usage of its functions on both the Kickstarter and Indiegogo datasets. Each tuned model is evaluated separately for each dataset (using the hyperparameters tuned specifically for that dataset).

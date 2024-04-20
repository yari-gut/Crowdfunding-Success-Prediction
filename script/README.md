# Data Preprocessing

The `data_preprocessing.py` script performs preprocessing tasks on Kickstarter and Indiegogo datasets before merging them. The preprocessing steps include:

- **Kickstarter Data Preprocessing:**
  - Drop unnecessary columns such as 'ID', 'Subcategory', and 'Country'.
  - Convert 'Launched' and 'Deadline' columns to datetime format.
  - Calculate the campaign duration in days from 'Launched' and 'Deadline' columns.
  - Convert the 'State' column to binary, representing Successful (1) or Failed (0) campaigns.
  - Normalize numerical features ('Goal', 'Pledged', 'Backers') using Min-Max scaling.
  - Drop rows with missing values.

- **Indiegogo Data Preprocessing:**
  - Drop unnecessary columns such as 'currency', 'year_end', 'month_end', 'day_end', 'time_end', 'project_id', 'tagline', 'url', 'date_launch', and 'date_end'.
  - Convert 'year_launch', 'month_launch', 'day_launch', 'time_launch' columns to datetime format.
  - Normalize numerical features ('amount_raised_usd', 'goal_usd') using Min-Max scaling.
  - Drop rows with missing values.

- **Aligning Column Names:**
  - Align column names between Indiegogo and Kickstarter datasets by mapping them to a common format.

- **Merging Datasets:**
  - Merge the preprocessed Kickstarter and Indiegogo datasets into a single dataset.
  - Save the merged dataset to a CSV file named 'merged_dataset.csv'.

The script ensures that both datasets are properly preprocessed and aligned before merging, facilitating further analysis and model training in subsequent stages of the project.

## Model Tuning

The `model_tuning.py` script tunes hyperparameters for various machine learning classifiers using Grid Search CV. It performs the following tasks:

- **Split Data:** The script splits the input dataset into training and testing sets using an 80-20 ratio.

- **Classifier Selection:** It defines a dictionary of classifiers, each paired with a parameter grid for hyperparameter tuning. The supported classifiers include:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
  - Naive Bayes

- **Hyperparameter Tuning:** For each classifier, the script utilizes Grid Search CV to find the optimal combination of hyperparameters. It searches through the specified parameter grid and evaluates each combination using cross-validation.

- **Best Parameters:** After hyperparameter tuning, the script identifies the best parameters for each classifier based on the highest cross-validation accuracy.

- **Model Training:** Using the best parameters, the script trains each classifier on the training set. It fits the classifier to the training data, effectively learning from the features and labels.

- **Model Evaluation:** Once trained, each classifier is evaluated on the testing set to assess its performance. The script computes the accuracy of each classifier, indicating the proportion of correctly classified instances.

- **Results Storage:** The script organizes the results, including the best parameters and accuracy for each classifier, into a JSON object.

The `model_tuning.py` script enables efficient optimization of machine learning classifiers by systematically searching for the best hyperparameters. It facilitates the selection of optimal models for predicting crowdfunding campaign success, enhancing the overall effectiveness of the project's predictive analytics.


# Model Evaluation



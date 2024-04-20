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

# Model Evaluation


# Model Training



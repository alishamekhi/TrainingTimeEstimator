# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import csv
def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):

    # Show more than 10 or 20 rows when a dataframe comes back.
    pd.set_option('display.max_rows', max_rows)
    # Columns displayed in debug view
    pd.set_option('display.max_columns', max_columns)
    pd.options.display.float_format = '{:.2f}'.format
    pd.set_option('display.width', display_width)


# run
pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320)

output_file = 'Results\\Estimator_Training_Log.csv'
StartTime = datetime.today()

data = pd.read_csv("DS\\TrainingTimeData.csv")

# Split the data into features (X) and target variable (y)
X = data.drop('TrainingTime', axis=1)
y = data['TrainingTime']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

EndTime = datetime.today()

with open(output_file, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    data_row = [StartTime.strftime("%Y-%m-%d_%H:%M:%S"),EndTime.strftime("%Y-%m-%d_%H:%M:%S"),"RF",mse]
    csv_writer.writerow(data_row)

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
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

# Initialize the XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = xg_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# You can also print feature importance
print("Feature Importance:")
for feature, importance in zip(X.columns, xg_reg.feature_importances_):
    print(f"{feature}: {importance}")


EndTime = datetime.today()

with open(output_file, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    data_row = [StartTime.strftime("%Y-%m-%d_%H:%M:%S"),EndTime.strftime("%Y-%m-%d_%H:%M:%S"),"XGBoost",mse]
    csv_writer.writerow(data_row)

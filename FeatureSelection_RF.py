from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
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


# Create a RandomForestRegressor for regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to your data
rf_model.fit(X, y)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to show feature names and their importance scores
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature',fontsize=18)
plt.ylabel('Importance Score',fontsize=18)
plt.title('RandomForestRegressor Feature Importance (Ordered)',fontsize=22)
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
# Increase the size of the axis points
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Optionally, you can print the feature importance DataFrame
print(feature_importance_df)
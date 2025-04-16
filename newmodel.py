# Linear model with sea temp, wave power, period and height averages for each year


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ndvi_file = "D:/FYP/Data/updated_data.csv"
output_file = "D:/FYP/Data/predicted_NDVI_2024.csv"
data = pd.read_csv(ndvi_file)

# Replace null values in FID (sediment type) with 0 (water)
data['FID'].fillna(0, inplace=True)

# nulls replaced with means
data.fillna(data.mean(), inplace=True)


# using all data (excluding 2024) to predict NDVI_2024H
features = data.drop(columns=[
    'OBJECTID', 'Longitude', 'Latitude', 'FID',
    'NDVI_2024H', 'NDVI_2024L'  # Exclude the target columns
])

target_H = data['NDVI_2024H']
target_L = data['NDVI_2024L']

# Split data into training and testing sets
X_train_H, X_test_H, y_train_H, y_test_H = train_test_split(features, target_H, test_size=0.2, random_state=42)
X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(features, target_L, test_size=0.2, random_state=42)

# Model for NDVI_2024H
print("\nTraining model for NDVI_2024H...")
with tqdm(total=3, desc="NDVI_2024H", bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:
    model_H = LinearRegression()
    pbar.update(1)

    model_H.fit(X_train_H, y_train_H)
    pbar.update(1)

    predictions_H = model_H.predict(features)  # Predict for all coordinates
    pbar.update(1)

# Model for NDVI_2024L
print("\nTraining model for NDVI_2024L...")
with tqdm(total=3, desc="NDVI_2024L", bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:
    model_L = LinearRegression()
    pbar.update(1)

    model_L.fit(X_train_L, y_train_L)
    pbar.update(1)

    predictions_L = model_L.predict(features)  # Predict for all coordinates
    pbar.update(1)

# Combine predictions with coordinates
output_df = data[['OBJECTID', 'Longitude', 'Latitude', 'FID']].copy()
output_df['Predicted_NDVI_2024H'] = predictions_H
output_df['Predicted_NDVI_2024L'] = predictions_L

# save to csv
output_df.to_csv(output_file, index=False)

# r squared and mse
print("\nNDVI_2024H Model Performance:")
print("Mean Squared Error:", mean_squared_error(y_test_H, model_H.predict(X_test_H)))
print("R2 Score:", r2_score(y_test_H, model_H.predict(X_test_H)))

print("\nNDVI_2024L Model Performance:")
print("Mean Squared Error:", mean_squared_error(y_test_L, model_L.predict(X_test_L)))
print("R2 Score:", r2_score(y_test_L, model_L.predict(X_test_L)))

print("\nPredictions saved to 'predicted_NDVI_2024.csv'")


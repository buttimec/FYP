import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Option to enable outlier detection and plotting
enable_outliers = True
threshhold = 7  # for NDVI_2024H; for NDVI_2024L a multiplier of 7 is used to identify the extreme outliers

data = pd.read_csv("D:/FYP/Data/XGBoost data.csv")
data = data.drop(columns=['OBJECTID', 'FID', 'DESCRIPT'])

# cluster labels (0–6) given more descriptive names (identified these through the cluster analysis)
cluster_mapping = {
    0: 'Water',
    1: 'Land - Limestone',
    2: 'Water 2',
    3: 'Land 2',
    4: 'Water 3',
    5: 'Water - Tidal',
    6: 'Tide Vary'
}
# ensure cluster is integer and create a new column for cluster names (for display/graphing)
data['Cluster'] = data['Cluster'].astype(int)
data['ClusterName'] = data['Cluster'].map(cluster_mapping)

#replace nulls with averages
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

#null sediment labelled as water
data['STRATCODE'] = data['STRATCODE'].fillna('WATER')
data['STRATCODE'] = data['STRATCODE'].replace('', 'WATER')


# Model for NDVI_2024H Prediction (High Tide) - (GenAI assisted in debugging)

target_H = data['NDVI_2024H']
features_H = data.drop(columns=['NDVI_2024H'])

features_H = pd.get_dummies(features_H, columns=['STRATCODE', 'Cluster'], drop_first=True)
if 'ClusterName' in features_H.columns:
    features_H = features_H.drop(columns=['ClusterName'])
#training and testing
X_train_H, X_test_H, y_train_H, y_test_H, train_idx, test_idx = train_test_split(
    features_H, target_H, data.index, test_size=0.2, random_state=42
)

# Train XGBoost regressor for NDVI_2024H - based off literature/CPF cluster and GenAI error corrections
xgb_reg_H = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression with squared error loss
    learning_rate=0.1,             # Step size shrinkage
    max_depth=5,                   # Max tree depth
    n_estimators=100,              # Number of boosting rounds
    random_state=42                # Ensures reproducibility
)
xgb_reg_H.fit(X_train_H, y_train_H)

y_pred_H = xgb_reg_H.predict(X_test_H)
mse_H = mean_squared_error(y_test_H, y_pred_H)
r2_H = r2_score(y_test_H, y_pred_H)
mae_H = mean_absolute_error(y_test_H, y_pred_H)
explained_var_H = explained_variance_score(y_test_H, y_pred_H)

print("Evaluation Metrics for NDVI_2024H:")
print(f"MSE: {mse_H:.4f}")
print(f"R2: {r2_H:.4f}")
print(f"MAE: {mae_H:.4f}")
print(f"Explained Variance: {explained_var_H:.4f}")

clusters_to_graph = None  # Set to None to display all clusters - was used to analyse different
# clusters as the land/water clusters were much larger than the others and sometimes made it impossible to view the others.
test_data_H = data.loc[test_idx].copy()
test_data_H['Predicted_NDVI_2024H'] = y_pred_H

if enable_outliers:
    # Compute residuals for NDVI_2024H
    test_data_H['Residual_H'] = abs(test_data_H['NDVI_2024H'] - test_data_H['Predicted_NDVI_2024H'])
    # Use a threshold: mean + threshhold * std (more selective, fewer outliers)
    threshold_H = test_data_H['Residual_H'].mean() + threshhold * test_data_H['Residual_H'].std()
    outliers_H = test_data_H[test_data_H['Residual_H'] > threshold_H]
    print("\nMajor Outliers for NDVI_2024H:")
    if not outliers_H.empty:
        print(outliers_H[['NDVI_2024H', 'Predicted_NDVI_2024H', 'Residual_H', 'ClusterName']])
        # Save the high tide outliers to CSV
        outliers_H.to_csv("D:/FYP/Data/outliers_NDVI_2024H.csv", index=False)
        print("NDVI_2024H outliers saved to 'D:/FYP/Data/outliers_NDVI_2024H.csv'")
    else:
        print("No major outliers detected.")
else:
    outliers_H = pd.DataFrame()

plt.figure(figsize=(10, 6))
if clusters_to_graph is None:
    clusters_to_graph = test_data_H['ClusterName'].unique()
for cluster in clusters_to_graph:
    subset = test_data_H[test_data_H['ClusterName'] == cluster]
    if not subset.empty:
        plt.scatter(subset['NDVI_2024H'], subset['Predicted_NDVI_2024H'],
                    label=cluster, alpha=0.7)
# Highlight outliers if enabled
if enable_outliers and not outliers_H.empty:
    plt.scatter(outliers_H['NDVI_2024H'], outliers_H['Predicted_NDVI_2024H'],
                facecolors='none', edgecolors='red', s=100, label='Outliers')
min_val_H = min(test_data_H['NDVI_2024H'].min(), test_data_H['Predicted_NDVI_2024H'].min())
max_val_H = max(test_data_H['NDVI_2024H'].max(), test_data_H['Predicted_NDVI_2024H'].max())
plt.plot([min_val_H, max_val_H], [min_val_H, max_val_H], 'k--', lw=2)
plt.xlabel("Actual NDVI_2024H")
plt.ylabel("Predicted NDVI_2024H")
plt.title("Actual vs. Predicted NDVI_2024H by Cluster")
plt.legend(title="Cluster")
plt.show()

# Model for NDVI_2024L Prediction (Low Tide) same as above

# Define target and features for NDVI_2024L
target_L = data['NDVI_2024L']
features_L = data.drop(columns=['NDVI_2024L'])

features_L = pd.get_dummies(features_L, columns=['STRATCODE', 'Cluster'], drop_first=True)
if 'ClusterName' in features_L.columns:
    features_L = features_L.drop(columns=['ClusterName'])

# Use the same test indices
train_idx_L = data.index.difference(test_idx)
X_train_L = features_L.loc[train_idx_L]
y_train_L = target_L.loc[train_idx_L]
X_test_L = features_L.loc[test_idx]
y_test_L = target_L.loc[test_idx]

# Train XGBoost regressor for NDVI_2024L same as high tide (genAi assissted like above)
xgb_reg_L = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    random_state=42
)
xgb_reg_L.fit(X_train_L, y_train_L)

y_pred_L = xgb_reg_L.predict(X_test_L)
mse_L = mean_squared_error(y_test_L, y_pred_L)
r2_L = r2_score(y_test_L, y_pred_L)
mae_L = mean_absolute_error(y_test_L, y_pred_L)
explained_var_L = explained_variance_score(y_test_L, y_pred_L)

print("\nEvaluation Metrics for NDVI_2024L:")
print(f"MSE: {mse_L:.4f}")
print(f"R2: {r2_L:.4f}")
print(f"MAE: {mae_L:.4f}")
print(f"Explained Variance: {explained_var_L:.4f}")

test_data_L = data.loc[test_idx].copy()
test_data_L['Predicted_NDVI_2024L'] = y_pred_L

if enable_outliers:
    # Compute residuals for NDVI_2024L
    test_data_L['Residual_L'] = abs(test_data_L['NDVI_2024L'] - test_data_L['Predicted_NDVI_2024L'])
    threshold_L = test_data_L['Residual_L'].mean() + 3 * test_data_L['Residual_L'].std()
    outliers_L = test_data_L[test_data_L['Residual_L'] > threshold_L]
    print("\nMajor Outliers for NDVI_2024L:")
    if not outliers_L.empty:
        print(outliers_L[['NDVI_2024L', 'Predicted_NDVI_2024L', 'Residual_L', 'ClusterName']])
        # saved to csv to examine them
        outliers_L.to_csv("D:/FYP/Data/outliers_NDVI_2024L.csv", index=False)
        print("NDVI_2024L outliers saved to 'D:/FYP/Data/outliers_NDVI_2024L.csv'")
    else:
        print("No major outliers detected.")
else:
    outliers_L = pd.DataFrame()

plt.figure(figsize=(10, 6))
if clusters_to_graph is None:
    clusters_to_graph = test_data_L['ClusterName'].unique()
for cluster in clusters_to_graph:
    subset = test_data_L[test_data_L['ClusterName'] == cluster]
    if not subset.empty:
        plt.scatter(subset['NDVI_2024L'], subset['Predicted_NDVI_2024L'],
                    label=cluster, alpha=0.7)
# Highlight outliers if enabled
if enable_outliers and not outliers_L.empty:
    plt.scatter(outliers_L['NDVI_2024L'], outliers_L['Predicted_NDVI_2024L'],
                facecolors='none', edgecolors='red', s=100, label='Outliers')
min_val_L = min(test_data_L['NDVI_2024L'].min(), test_data_L['Predicted_NDVI_2024L'].min())
max_val_L = max(test_data_L['NDVI_2024L'].max(), test_data_L['Predicted_NDVI_2024L'].max())
plt.plot([min_val_L, max_val_L], [min_val_L, max_val_L], 'k--', lw=2)
plt.xlabel("Actual NDVI_2024L")
plt.ylabel("Predicted NDVI_2024L")
plt.title("Actual vs. Predicted NDVI_2024L by Cluster")
plt.legend(title="Cluster")
plt.show()

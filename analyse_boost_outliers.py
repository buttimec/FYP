# Code used to graph and analyse the extreme outliers.
# GenAI was used to debug errors and suggest additional ways to represent the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

enable_outliers = True  # Set to False to disable outlier-related analyses
threshhold = 7  #

# Define file paths for the outlier CSV files
file_path_H = "D:/FYP/Data/outliers_NDVI_2024H.csv"
file_path_L = "D:/FYP/Data/outliers_NDVI_2024L.csv"
full_data_path = "D:/FYP/Data/XGBoost data.csv"  # Full original dataset

try:
    outliers_H = pd.read_csv(file_path_H)
    outliers_L = pd.read_csv(file_path_L)
    print("High tide outliers loaded:", outliers_H.shape)
    print("Low tide outliers loaded:", outliers_L.shape)
except Exception as e:
    print("Error loading outlier files:", e)
    exit()

try:
    full_data = pd.read_csv(full_data_path)
    print("Full dataset loaded:", full_data.shape)
except Exception as e:
    print("Error loading full dataset:", e)
    exit()


def print_descriptive_stats(df, tide_type="High Tide"):
    print(f"\nDescriptive Statistics for {tide_type}:")
    if tide_type == "High Tide":
        cols = ['NDVI_2024H', 'Predicted_NDVI_2024H', 'Residual_H']
    else:
        cols = ['NDVI_2024L', 'Predicted_NDVI_2024L', 'Residual_L']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe())

print_descriptive_stats(outliers_H, "High Tide")
print_descriptive_stats(outliers_L, "Low Tide")


def plot_residuals_hist_box(df, tide_type="High Tide"):
    resid_col = 'Residual_H' if tide_type == "High Tide" else 'Residual_L'
    if resid_col not in df.columns:
        print(f"No {resid_col} column found for {tide_type}.")
        return

    plt.figure(figsize=(12,5))
    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(df[resid_col], bins=20, kde=True, color='skyblue')
    plt.title(f"{tide_type} Residuals Histogram")
    plt.xlabel("Absolute Residual")

    # Box plot
    plt.subplot(1,2,2)
    sns.boxplot(x=df[resid_col], color='lightgreen')
    plt.title(f"{tide_type} Residuals Boxplot")
    plt.xlabel("Absolute Residual")

    plt.tight_layout()
    plt.show()

plot_residuals_hist_box(outliers_H, "High Tide")
plot_residuals_hist_box(outliers_L, "Low Tide")

# GenAI assisted scatter plots for actual vs predicted values

def scatter_actual_vs_pred(df, tide_type="High Tide"):
    if tide_type == "High Tide":
        actual, pred = 'NDVI_2024H', 'Predicted_NDVI_2024H'
    else:
        actual, pred = 'NDVI_2024L', 'Predicted_NDVI_2024L'

    if actual not in df.columns or pred not in df.columns:
        print(f"Required columns missing for {tide_type} scatter plot.")
        return

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df[actual], y=df[pred], hue=df.get('ClusterName'), palette="tab10", s=60)
    min_val = min(df[actual].min(), df[pred].min())
    max_val = max(df[actual].max(), df[pred].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.title(f"Actual vs. Predicted {tide_type} NDVI")
    plt.xlabel("Actual NDVI")
    plt.ylabel("Predicted NDVI")
    if 'ClusterName' in df.columns:
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

scatter_actual_vs_pred(outliers_H, "High Tide")
scatter_actual_vs_pred(outliers_L, "Low Tide")

cols_corr_H = [
    'NDVI_2016H', 'NDVI_2017H', 'NDVI_2018H', 'NDVI_2019H',
    'NDVI_2020H', 'NDVI_2021H', 'NDVI_2022H', 'NDVI_2023H',
    'NDVI_2024H', 'Predicted_NDVI_2024H'
]
cols_corr_L = [
    'NDVI_2016L', 'NDVI_2017L', 'NDVI_2018L', 'NDVI_2019L',
    'NDVI_2020L', 'NDVI_2021L', 'NDVI_2022L', 'NDVI_2023L',
    'NDVI_2024L', 'Predicted_NDVI_2024L'
]

def plot_corr_matrix(df, cols, title="Correlation Matrix"):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"No valid columns for {title}.")
        return
    corr = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_corr_matrix(outliers_H, cols_corr_H, "Correlation Matrix for High Tide Outlier Variables")
plot_corr_matrix(outliers_L, cols_corr_L, "Correlation Matrix for Low Tide Outlier Variables")

# statistical tests code supported by GenAI
def perform_pearson_test(df, col1, col2, tide_type="High Tide"):
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Columns {col1} or {col2} missing in {tide_type} data.")
        return
    corr, p_val = stats.pearsonr(df[col1], df[col2])
    print(f"Pearson correlation between {col1} and {col2} for {tide_type}: {corr:.4f} (p={p_val:.4f})")

if 'Residual_H' in outliers_H.columns and 'NDVI_2024H' in outliers_H.columns:
    perform_pearson_test(outliers_H, 'Residual_H', 'NDVI_2024H', "High Tide")
if 'Residual_L' in outliers_L.columns and 'NDVI_2024L' in outliers_L.columns:
    perform_pearson_test(outliers_L, 'Residual_L', 'NDVI_2024L', "Low Tide")

def ttest_between_clusters(df, cluster1, cluster2, residual_col, tide_type="High Tide"):
    data1 = df[df['ClusterName'] == cluster1][residual_col]
    data2 = df[df['ClusterName'] == cluster2][residual_col]
    if len(data1)==0 or len(data2)==0:
        print(f"Not enough data for t-test between {cluster1} and {cluster2} in {tide_type}.")
        return
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
    print(f"T-test for {tide_type} residuals between {cluster1} and {cluster2}: t={t_stat:.4f}, p={p_val:.4f}")

if 'ClusterName' in outliers_H.columns:
    ttest_between_clusters(outliers_H, "Water", "Land - Limestone", "Residual_H", "High Tide")
if 'ClusterName' in outliers_L.columns:
    ttest_between_clusters(outliers_L, "Water", "Land - Limestone", "Residual_L", "Low Tide")


plt.figure(figsize=(10,6))
# high tide outliers more stand out
if not outliers_H.empty and 'Longitude' in outliers_H.columns and 'Latitude' in outliers_H.columns:
    plt.scatter(
        outliers_H['Longitude'],
        outliers_H['Latitude'],
        c='red',            # Red color
        marker='o',         # Circular marker
        s=150,              # Larger size
        alpha=1.0,          # Fully opaque
        label="High Tide Outliers"
    )
# Plot Low Tide Outliers: smaller markers, partially transparent
if not outliers_L.empty and 'Longitude' in outliers_L.columns and 'Latitude' in outliers_L.columns:
    plt.scatter(
        outliers_L['Longitude'],
        outliers_L['Latitude'],
        c='blue',           # Blue color
        marker='x',         # 'X' marker
        s=80,               # Slightly smaller size
        alpha=0.3,          # More transparent
        label="Low Tide Outliers"
    )
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Outliers")
plt.legend()
plt.tight_layout()
plt.show()


def pairplot_variables(df, cols, title="Pairplot"):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print(f"No valid columns for pairplot: {title}")
        return
    sns.pairplot(df[cols], diag_kind="kde")
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

subset_cols_H = ['NDVI_2024H', 'Predicted_NDVI_2024H', 'Residual_H', 'NDVI_2021H', 'WaveHeight2016']
subset_cols_L = ['NDVI_2024L', 'Predicted_NDVI_2024L', 'Residual_L', 'NDVI_2021L', 'SeaTemp2016']
if not outliers_H.empty:
    pairplot_variables(outliers_H, subset_cols_H, "Pairplot for Selected High Tide Outlier Variables")
if not outliers_L.empty:
    pairplot_variables(outliers_L, subset_cols_L, "Pairplot for Selected Low Tide Outlier Variables")

# ANOVA / Group Comparisons by Cluster

def anova_by_cluster(df, residual_col, tide_type="High Tide"):
    if residual_col not in df.columns or 'ClusterName' not in df.columns:
        print(f"Missing columns for ANOVA in {tide_type}.")
        return
    groups = [group[residual_col].values for name, group in df.groupby('ClusterName')]
    if len(groups) < 2:
        print(f"Not enough clusters for ANOVA in {tide_type}.")
        return
    anova_res = stats.f_oneway(*groups)
    print(f"\nANOVA result for {tide_type} residuals by Cluster: F={anova_res.statistic:.4f}, p={anova_res.pvalue:.4f}")

if 'Residual_H' in outliers_H.columns and 'ClusterName' in outliers_H.columns:
    anova_by_cluster(outliers_H, 'Residual_H', "High Tide")
if 'Residual_L' in outliers_L.columns and 'ClusterName' in outliers_L.columns:
    anova_by_cluster(outliers_L, 'Residual_L', "Low Tide")

def shapiro_and_qq(df, residual_col, tide_type="High Tide"):
    if residual_col not in df.columns:
        print(f"No column {residual_col} for {tide_type} analysis.")
        return
    stat, p = stats.shapiro(df[residual_col])
    print(f"Shapiro-Wilk test for {tide_type} {residual_col}: stat={stat:.4f}, p={p:.4f}")
    plt.figure(figsize=(6,6))
    stats.probplot(df[residual_col], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot for {tide_type} {residual_col}")
    plt.tight_layout()
    plt.show()

if enable_outliers:
    shapiro_and_qq(outliers_H, 'Residual_H', "High Tide")
    shapiro_and_qq(outliers_L, 'Residual_L', "Low Tide")

def barplot_mean_residual_by_cluster(df, residual_col, tide_type="High Tide"):
    if 'ClusterName' not in df.columns:
        print(f"ClusterName column missing in {tide_type} data.")
        return
    grouped = df.groupby('ClusterName')[residual_col].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(8,6))
    ax = sns.barplot(x='ClusterName', y='mean', data=grouped, palette="viridis", capsize=0.1)
    for i, row in grouped.iterrows():
        ax.errorbar(i, row['mean'], yerr=row['std'], fmt='none', c='black')
    plt.title(f"Mean {residual_col} by Cluster ({tide_type})")
    plt.xlabel("Cluster")
    plt.ylabel(f"Mean {residual_col}")
    plt.tight_layout()
    plt.show()

if enable_outliers:
    if 'Residual_H' in outliers_H.columns:
        barplot_mean_residual_by_cluster(outliers_H, 'Residual_H', "High Tide")
    if 'Residual_L' in outliers_L.columns:
        barplot_mean_residual_by_cluster(outliers_L, 'Residual_L', "Low Tide")

if enable_outliers:
    # For High Tide Outliers: Use full_data to capture all underlying variables
    full_data = pd.read_csv(full_data_path)  # Reload the full dataset
    test_data_H = full_data.loc[outliers_H.index].copy()  # Select rows corresponding to high tide outliers
    if 'Residual_H' in test_data_H.columns:
        Q1_H = test_data_H['Residual_H'].quantile(0.25)
        Q3_H = test_data_H['Residual_H'].quantile(0.75)
        IQR_H = Q3_H - Q1_H
        extreme_threshold_H = Q3_H + 1.5 * IQR_H
        extreme_outliers_H = test_data_H[test_data_H['Residual_H'] > extreme_threshold_H]
        if not extreme_outliers_H.empty:
            extreme_outliers_H.to_csv("D:/FYP/Data/extreme_outliers_NDVI_2024H.csv", index=False)
            print("Extreme high tide outliers saved to 'D:/FYP/Data/extreme_outliers_NDVI_2024H.csv'")
        else:
            print("No extreme high tide outliers detected by boxplot rule.")

    # For Low Tide Outliers: Use full_data similarly
    test_data_L = full_data.loc[outliers_L.index].copy()
    if 'Residual_L' in test_data_L.columns:
        Q1_L = test_data_L['Residual_L'].quantile(0.25)
        Q3_L = test_data_L['Residual_L'].quantile(0.75)
        IQR_L = Q3_L - Q1_L
        extreme_threshold_L = Q3_L + 1.5 * IQR_L
        extreme_outliers_L = test_data_L[test_data_L['Residual_L'] > extreme_threshold_L]
        if not extreme_outliers_L.empty:
            extreme_outliers_L.to_csv("D:/FYP/Data/extreme_outliers_NDVI_2024L.csv", index=False)
            print("Extreme low tide outliers saved to 'D:/FYP/Data/extreme_outliers_NDVI_2024L.csv'")
        else:
            print("No extreme low tide outliers detected by boxplot rule.")

print("\nAnalysis complete. Please review the generated plots and statistics to investigate why these points may be outliers.")

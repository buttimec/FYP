
"""
plot_geographical_outliers_separate.py - (GenAI was used to solve errors and make suggestions)

This script loads the outlier CSV files for NDVI_2024H and NDVI_2024L,
then filters the data so that for high tide, points with Residual_H > 0.3 are
plotted in dark red while the others in pink; for low tide, points with Residual_L > 0.25
are plotted in dark blue and the others in light blue. It also graphs the NDVI time series
for each major residual point (Residual_H > 0.3 or Residual_L > 0.25) from 2016 to 2024 (actual)
with the predicted NDVI value plotted at 2025 (labeled as "2024 (predicted)").
Additionally, it displays the NDVI time series for the other residual points (below the thresholds)
in separate graphs. For the low tide "others" graph, you can choose to display either all points
or a random 20% subset.
Each time series is color‑coded by its cluster number.

These plots show the points according to their coordinates (Longitude and Latitude),
color‑coded based on NDVI accuracy. For low tide, points where the actual 2024 NDVI is lower
than predicted (indicating erosion) are colored red, and points with actual greater than predicted
are green. For high tide, the colors are swapped: points where the actual 2024 NDVI is greater
than predicted (erosion for high tide) are colored red, while those where it is lower (accretion)
are colored green. Separate plots are provided for high tide, for low tide, and a combined plot.

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

# Define file paths for the outlier CSV files
file_path_H = "D:/FYP/Data/outliers_NDVI_2024H.csv"
file_path_L = "D:/FYP/Data/outliers_NDVI_2024L.csv"

# Load the outlier data
try:
    df_high = pd.read_csv(file_path_H)
    df_low  = pd.read_csv(file_path_L)
    print("High tide outliers loaded:", df_high.shape)
    print("Low tide outliers loaded:", df_low.shape)
except Exception as e:
    print("Error loading files:", e)
    exit()

# Set thresholds for extreme points (major residuals)
threshold_high = 0.3  # For high tide: Residual_H > 0.3
threshold_low  = 0.25  # For low tide:  Residual_L > 0.25

# Filter high tide points: important vs. others
high_important = df_high[df_high['Residual_H'] > threshold_high]
high_other = df_high[df_high['Residual_H'] <= threshold_high]

# Filter low tide points: important vs. others
low_important = df_low[df_low['Residual_L'] > threshold_low]
low_other = df_low[df_low['Residual_L'] <= threshold_low]

#high tide
plt.figure(figsize=(10,6))
plt.scatter(high_other['Longitude'], high_other['Latitude'],
            c='pink', marker='o', s=80, alpha=0.7,
            label="High Tide (Residual ≤ 0.3)")
plt.scatter(high_important['Longitude'], high_important['Latitude'],
            c='darkred', marker='o', s=150, alpha=1.0,
            label="High Tide (Residual > 0.3)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("High Tide Outlier Locations")
plt.legend()
plt.tight_layout()
plt.show()

#low tide
plt.figure(figsize=(10,6))
plt.scatter(low_other['Longitude'], low_other['Latitude'],
            c='lightblue', marker='x', s=80, alpha=0.7,
            label="Low Tide (Residual ≤ 0.25)")
plt.scatter(low_important['Longitude'], low_important['Latitude'],
            c='darkblue', marker='x', s=150, alpha=1.0,
            label="Low Tide (Residual > 0.25)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Low Tide Outlier Locations")
plt.legend()
plt.tight_layout()
plt.show()

#low and high tide
plt.figure(figsize=(10,6))
# high tide points
plt.scatter(high_other['Longitude'], high_other['Latitude'],
            c='pink', marker='o', s=80, alpha=0.7,
            label="High Tide (Residual ≤ 0.3)")
plt.scatter(high_important['Longitude'], high_important['Latitude'],
            c='darkred', marker='o', s=150, alpha=1.0,
            label="High Tide (Residual > 0.3)")
# low tide points
plt.scatter(low_other['Longitude'], low_other['Latitude'],
            c='lightblue', marker='x', s=80, alpha=0.7,
            label="Low Tide (Residual ≤ 0.25)")
plt.scatter(low_important['Longitude'], low_important['Latitude'],
            c='darkblue', marker='x', s=150, alpha=1.0,
            label="Low Tide (Residual > 0.25)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Combined Outlier Locations (High & Low Tide)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# NDVI Time Series for High Tide Points (GenAI assisted)

plt.figure(figsize=(10,6))
predicted_year = 2025  # Predicted NDVI plotted at 2025 for equal spacing

clusters_high = sorted(high_important['Cluster'].unique())
colormap_high = cm.get_cmap('tab20', len(clusters_high))
colors_high = {cluster: colormap_high(i) for i, cluster in enumerate(clusters_high)}

for idx, row in high_important.iterrows():
    x_measured = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    measured = [row["NDVI_2016H"], row["NDVI_2017H"], row["NDVI_2018H"], row["NDVI_2019H"],
                row["NDVI_2020H"], row["NDVI_2021H"], row["NDVI_2022H"], row["NDVI_2023H"],
                row["NDVI_2024H"]]  # Actual 2024 measured value
    predicted = row["Predicted_NDVI_2024H"]
    cluster = row["Cluster"]
    color = colors_high.get(cluster, 'blue')

    plt.plot(x_measured, measured, marker='o', linestyle='-', alpha=0.7, color=color)
    plt.plot([2024, predicted_year], [measured[-1], predicted], marker='o', linestyle='--', alpha=0.7, color=color)
    plt.scatter(predicted_year, predicted, marker='D', color=color, s=60, zorder=5)

legend_elements_high = [plt.Line2D([0], [0], color=colors_high[cluster], lw=2, label=f'Cluster {cluster}')
                        for cluster in clusters_high]
plt.legend(handles=legend_elements_high, loc='best')
plt.xlabel("Year")
plt.ylabel("NDVI (High Tide)")
plt.title("NDVI Time Series for High Tide Points\n(2016-2024 Actual, 2024 (predicted))")
plt.xticks([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, predicted_year],
           ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024 (actual)", "2024 (predicted)"],
           rotation=45)
plt.tight_layout()
plt.show()


#  NDVI Time Series for Low Tide Points (GenAI assisted debugging errors)

plt.figure(figsize=(10,6))
predicted_year = 2025

clusters_low = sorted(low_important['Cluster'].unique())
colormap_low = cm.get_cmap('tab20', len(clusters_low))
colors_low = {cluster: colormap_low(i) for i, cluster in enumerate(clusters_low)}

for idx, row in low_important.iterrows():
    x_measured = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    measured = [row["NDVI_2016L"], row["NDVI_2017L"], row["NDVI_2018L"], row["NDVI_2019L"],
                row["NDVI_2020L"], row["NDVI_2021L"], row["NDVI_2022L"], row["NDVI_2023L"],
                row["NDVI_2024L"]]  # Actual 2024 measured value
    predicted = row["Predicted_NDVI_2024L"]
    cluster = row["Cluster"]
    color = colors_low.get(cluster, 'green')

    plt.plot(x_measured, measured, marker='o', linestyle='-', alpha=0.7, color=color)
    plt.plot([2024, predicted_year], [measured[-1], predicted], marker='o', linestyle='--', alpha=0.7, color=color)
    plt.scatter(predicted_year, predicted, marker='D', color=color, s=60, zorder=5)

legend_elements_low = [plt.Line2D([0], [0], color=colors_low[cluster], lw=2, label=f'Cluster {cluster}')
                       for cluster in clusters_low]
plt.legend(handles=legend_elements_low, loc='best')
plt.xlabel("Year")
plt.ylabel("NDVI (Low Tide)")
plt.title("NDVI Time Series for Low Tide Points\n(2016-2024 Actual, 2024 (predicted))")
plt.xticks([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, predicted_year],
           ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024 (actual)", "2024 (predicted)"],
           rotation=45)
plt.tight_layout()
plt.show()

#------
plt.figure(figsize=(10,6))
predicted_year = 2025

clusters_high_other = sorted(high_other['Cluster'].unique())
colormap_high_other = cm.get_cmap('tab20', len(clusters_high_other))
colors_high_other = {cluster: colormap_high_other(i) for i, cluster in enumerate(clusters_high_other)}

for idx, row in high_other.iterrows():
    x_measured = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    measured = [row["NDVI_2016H"], row["NDVI_2017H"], row["NDVI_2018H"], row["NDVI_2019H"],
                row["NDVI_2020H"], row["NDVI_2021H"], row["NDVI_2022H"], row["NDVI_2023H"],
                row["NDVI_2024H"]]
    predicted = row["Predicted_NDVI_2024H"]
    cluster = row["Cluster"]
    color = colors_high_other.get(cluster, 'blue')

    plt.plot(x_measured, measured, marker='o', linestyle='-', alpha=0.7, color=color)
    plt.plot([2024, predicted_year], [measured[-1], predicted], marker='o', linestyle='--', alpha=0.7, color=color)
    plt.scatter(predicted_year, predicted, marker='D', color=color, s=60, zorder=5)

legend_elements_high_other = [plt.Line2D([0], [0], color=colors_high_other[cluster], lw=2, label=f'Cluster {cluster}')
                              for cluster in clusters_high_other]
plt.legend(handles=legend_elements_high_other, loc='best')
plt.xlabel("Year")
plt.ylabel("NDVI (High Tide)")
plt.title("NDVI Time Series for High Tide Points (Other Residuals)\n(2016-2024 Actual, 2024 (predicted))")
plt.xticks([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, predicted_year],
           ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024 (actual)", "2024 (predicted)"],
           rotation=45)
plt.tight_layout()
plt.show()

#NDVI Time Series for Low Tide Points (Other Residuals)

# ----------------------------
plt.figure(figsize=(10,6))
predicted_year = 2025

# Toggle this variable: True displays a random 20% subset; False displays all points (this had too many to make any proper
# conclusions).
display_random_subset = False

if display_random_subset:
    low_other_to_plot = low_other.sample(frac=0.2, random_state=42)
else:
    low_other_to_plot = low_other

clusters_low_other = sorted(low_other_to_plot['Cluster'].unique())
colormap_low_other = cm.get_cmap('tab20', len(clusters_low_other))
colors_low_other = {cluster: colormap_low_other(i) for i, cluster in enumerate(clusters_low_other)}

for idx, row in low_other_to_plot.iterrows():
    x_measured = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    measured = [row["NDVI_2016L"], row["NDVI_2017L"], row["NDVI_2018L"], row["NDVI_2019L"],
                row["NDVI_2020L"], row["NDVI_2021L"], row["NDVI_2022L"], row["NDVI_2023L"],
                row["NDVI_2024L"]]
    predicted = row["Predicted_NDVI_2024L"]
    cluster = row["Cluster"]
    color = colors_low_other.get(cluster, 'green')

    plt.plot(x_measured, measured, marker='o', linestyle='-', alpha=0.7, color=color)
    plt.plot([2024, predicted_year], [measured[-1], predicted], marker='o', linestyle='--', alpha=0.7, color=color)
    plt.scatter(predicted_year, predicted, marker='D', color=color, s=60, zorder=5)

legend_elements_low_other = [plt.Line2D([0], [0], color=colors_low_other[cluster], lw=2, label=f'Cluster {cluster}')
                             for cluster in clusters_low_other]
plt.legend(handles=legend_elements_low_other, loc='best')
plt.xlabel("Year")
plt.ylabel("NDVI (Low Tide)")
plt.title("NDVI Time Series for Low Tide Points (Other Residuals)\n(2016-2024 Actual, 2024 (predicted))\n" +
          ("Random 20% Sample" if display_random_subset else "All Points"))
plt.xticks([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, predicted_year],
           ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024 (actual)", "2024 (predicted)"],
           rotation=45)
plt.tight_layout()
plt.show()


#NDVI Accuracy Based on Actual vs Predicted Values
#Points are plotted according to coordinates and color‑coded based on accuracy.
# (GenAI assisted)

# For High Tide:
plt.figure(figsize=(10,6))
# For high tide, if actual > predicted, then erosion → red; if actual < predicted, then accretion → green.
high_erosion = df_high[df_high["NDVI_2024H"] > df_high["Predicted_NDVI_2024H"]]
high_accretion = df_high[df_high["NDVI_2024H"] < df_high["Predicted_NDVI_2024H"]]

plt.scatter(high_erosion['Longitude'], high_erosion['Latitude'],
            c='red', marker='o', s=80, alpha=0.7,
            label="High Tide: Erosion (Actual > Predicted)")
plt.scatter(high_accretion['Longitude'], high_accretion['Latitude'],
            c='green', marker='o', s=80, alpha=0.7,
            label="High Tide: Accretion (Actual < Predicted)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("High Tide NDVI Accuracy\n(2024 Actual vs Predicted)")
plt.legend()
plt.tight_layout()
plt.show()

# For Low Tide:
plt.figure(figsize=(10,6))
# For low tide, if actual < predicted, then erosion → red; if actual > predicted, then accretion → green.
low_erosion = df_low[df_low["NDVI_2024L"] < df_low["Predicted_NDVI_2024L"]]
low_accretion = df_low[df_low["NDVI_2024L"] > df_low["Predicted_NDVI_2024L"]]

plt.scatter(low_accretion['Longitude'], low_accretion['Latitude'],
            c='green', marker='x', s=80, alpha=0.7,
            label="Low Tide: Accretion (Actual > Predicted)")
plt.scatter(low_erosion['Longitude'], low_erosion['Latitude'],
            c='red', marker='x', s=80, alpha=0.7,
            label="Low Tide: Erosion (Actual < Predicted)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Low Tide NDVI Accuracy\n(2024 Actual vs Predicted)")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
# High tide points (swapped colors):
plt.scatter(high_erosion['Longitude'], high_erosion['Latitude'],
            c='green', marker='o', s=80, alpha=0.7,
            label="High Tide: Erosion (Actual > Predicted)")
plt.scatter(high_accretion['Longitude'], high_accretion['Latitude'],
            c='red', marker='o', s=80, alpha=0.7,
            label="High Tide: Accretion (Actual < Predicted)")
plt.scatter(low_accretion['Longitude'], low_accretion['Latitude'],
            c='green', marker='x', s=80, alpha=0.7,
            label="Low Tide: Accretion (Actual > Predicted)")
plt.scatter(low_erosion['Longitude'], low_erosion['Latitude'],
            c='red', marker='x', s=80, alpha=0.7,
            label="Low Tide: Erosion (Actual < Predicted)")

# Custom legend handles to differentiate tide types (GenAI based off my requirements and work)
high_handle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=8, label='High Tide: Erosion')
high_handle2 = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                             markersize=8, label='High Tide: Accretion')
low_handle = mlines.Line2D([], [], color='green', marker='x', linestyle='None',
                           markersize=8, label='Low Tide: Accretion')
low_handle2 = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                            markersize=8, label='Low Tide: Erosion')

plt.legend(handles=[high_handle, high_handle2, low_handle, low_handle2], loc='best')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Combined NDVI Accuracy\n(2024 Actual vs Predicted for High & Low Tide)")
plt.tight_layout()
plt.show()

import numpy as np
import seaborn as sns


# Outlier Density Heatmap with Coastline Outline (genAi suggested and helped)

combined_outliers = pd.concat([df_high, df_low])

plt.figure(figsize=(10,6))

sns.kdeplot(x=combined_outliers['Longitude'], y=combined_outliers['Latitude'],
            cmap="Reds", fill=True, levels=50, alpha=0.7)

plt.scatter(combined_outliers['Longitude'], combined_outliers['Latitude'],
            color='black', s=5, alpha=0.5, label="Outliers")

# # Coastline outline (replace with real coastline coordinates if available)
# min_lon, max_lon = combined_outliers['Longitude'].min(), combined_outliers['Longitude'].max()
# min_lat, max_lat = combined_outliers['Latitude'].min(), combined_outliers['Latitude'].max()
# coastline_lon = [min_lon, min_lon, max_lon, max_lon, min_lon]
# coastline_lat = [min_lat, max_lat, max_lat, min_lat, min_lat]

# plt.plot(coastline_lon, coastline_lat, color='black', linewidth=1.5, label="Coastline")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Outlier Density Heatmap")
plt.legend()
plt.tight_layout()
plt.show()

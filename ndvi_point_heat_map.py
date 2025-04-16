import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

#GenAI was used to debug errors and suggest a start to the heatmap

file_path = "D:/FYP/Data/normalized_ndvi_data_with_sediment.csv"
df = pd.read_csv(file_path)

# what points to plot
num_points = 50  # number of points to plot
start_index = 60  # index to start from
sample_ids = df['OBJECTID'].iloc[start_index:start_index + num_points]

years = list(range(2013, 2024))

tide_type = 'L'  # Change to L / H for low/High tide NDVI

ndvi_data = []

# Loop through sample points and collect NDVI values for each year and tide type
for sample_id in sample_ids:
    sample_df = df[df['OBJECTID'] == sample_id]

    ndvi_values = [] #selecting h or l value for that point
    for year in years:
        if tide_type == 'H':
            ndvi_value = sample_df[f'NDVI_{year}H'].values[0]
        elif tide_type == 'L':
            ndvi_value = sample_df[f'NDVI_{year}L'].values[0]
        ndvi_values.append(ndvi_value)

    ndvi_data.append(np.array(ndvi_values))

heatmap_df = pd.DataFrame(ndvi_data, columns=[f'{year}' for year in years], index=sample_ids)

# plot the heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(heatmap_df, cmap='YlGnBu', annot=False, cbar_kws={'label': 'NDVI Value'}, xticklabels=years, yticklabels=sample_ids)
plt.title(f"NDVI Heatmap for {tide_type} Tide (Points {start_index} to {start_index + num_points - 1})")
plt.xlabel("Year")
plt.ylabel("OBJECTID")
#plt.savefig(f"ndvi_heatmap_{tide_type}_{start_index}_{start_index + num_points - 1}.png")
plt.show()

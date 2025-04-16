import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#used to compare 2016 and 2024 ndvi values and highlight areas with the greatest decrease (erosion)


# Load the NDVI data
file_path = "D:/FYP/Data/normalized_ndvi_data_with_sediment.csv"
df = pd.read_csv(file_path)

# Select the NDVI columns for 2016-2023 (High and Low)
ndvi_columns = [
    'NDVI_2016H', 'NDVI_2016L',
    'NDVI_2017H', 'NDVI_2017L',
    'NDVI_2018H', 'NDVI_2018L',
    'NDVI_2019H', 'NDVI_2019L',
    'NDVI_2020H', 'NDVI_2020L',
    'NDVI_2021H', 'NDVI_2021L',
    'NDVI_2022H', 'NDVI_2022L',
    'NDVI_2023H', 'NDVI_2023L'
]

# Drop rows with null values
ndvi_data = df[ndvi_columns].dropna()

# Calculate the mean NDVI for 2016 and 2023 (combining high and low tide)
df['Mean_NDVI_2016'] = df[['NDVI_2016H', 'NDVI_2016L']].mean(axis=1)
df['Mean_NDVI_2023'] = df[['NDVI_2023H', 'NDVI_2023L']].mean(axis=1)

# Calculate the NDVI change for each point (2016-2023)
df['NDVI_Change'] = df['Mean_NDVI_2023'] - df['Mean_NDVI_2016']

# Erosion detection threshold
erosion_threshold = -0.3  # I found -0.3 was the best to show areas of significant erosion

# Identify erosion areas (where NDVI change is below the threshold)
erosion_areas = df[df['NDVI_Change'] < erosion_threshold]

print(f"Number of erosion areas detected: {len(erosion_areas)}")
print(erosion_areas[['Latitude', 'Longitude', 'NDVI_Change']])

# Plot erosion areas and coastline
plt.figure(figsize=(12, 8))

# plot the coastline by coordinate points along the coastline
unique_coastline = df.drop_duplicates(subset=['Longitude', 'Latitude'])

# used this to limit the number of points mapped as they became too clustered and overlapped to see any useful points
n = 50  # Select every 50th point
coastline_sampled = unique_coastline.iloc[::n]

# plot the sampled coastline as smaller black dots, just to see the outline of the coastline
plt.scatter(coastline_sampled['Longitude'], coastline_sampled['Latitude'],
            color='black', s=10, alpha=1.0, label='Coastline')

#plot erosion areas (as red points) on top of the coastline points
plt.scatter(erosion_areas['Longitude'], erosion_areas['Latitude'], c='red', alpha=0.6, label='Erosion Areas')

plt.title('Geographical Distribution of Erosion Areas with Coastline')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot NDVI change over time for erosion areas (only used 2016 onwards due to differences in the landsat ndvi values
ndvi_years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
plt.figure(figsize=(12, 8))

for idx, row in erosion_areas.iterrows():
    mean_ndvi = []
    for year in ndvi_years:
        high_column = f'NDVI_{year}H'
        low_column = f'NDVI_{year}L'
        mean_ndvi.append(row[[high_column, low_column]].mean())
    plt.plot(ndvi_years, mean_ndvi, label=f'Point {row["Latitude"]}, {row["Longitude"]}')

plt.title('NDVI Trends for Erosion Areas (2016-2023)')
plt.xlabel('Year')
plt.ylabel('Mean NDVI')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(True)
plt.show()

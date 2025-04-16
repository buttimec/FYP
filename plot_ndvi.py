import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

file_path = "D:/FYP/Data/normalized_ndvi_data_with_sediment.csv"
df = pd.read_csv(file_path)

sample_id = df['OBJECTID'].iloc[14-1]  # Example: First point (had to be -1 due to the format of the csv)
sample_df = df[df['OBJECTID'] == sample_id]

years = list(range(2013, 2023))
ndvi_h = [sample_df[f'NDVI_{year}H'].values[0] for year in years]
ndvi_l = [sample_df[f'NDVI_{year}L'].values[0] for year in years]

plt.figure(figsize=(10, 5))
plt.plot(years, ndvi_h, marker='o', label='High Tide NDVI')
plt.plot(years, ndvi_l, marker='s', label='Low Tide NDVI', linestyle='dashed')
plt.xlabel("Year")
plt.ylabel("NDVI Value")
plt.title(f"NDVI Trends for OBJECTID {sample_id}")
plt.legend()
plt.grid()
#plt.savefig("ndvi_plot.png")
plt.show()

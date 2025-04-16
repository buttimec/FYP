import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('Agg')  # Use non-GUI backend

# Load data
file_path = "D:/FYP/Data/normalized_ndvi_data_with_sediment.csv"
df = pd.read_csv(file_path)

num_points = 50  # number of points to plot
start_point = 0  # change index point
sample_ids = df['OBJECTID'].iloc[start_point:start_point + num_points]

years = list(range(2013, 2024))

tide_option = "low"  # Change this to "both" "high" or "low" depending on what plotting

plt.figure(figsize=(20, 12))

for sample_id in sample_ids:
    sample_df = df[df['OBJECTID'] == sample_id]

    if tide_option in ["high", "both"]:
        ndvi_h = [sample_df[f'NDVI_{year}H'].values[0] for year in years]
        plt.plot(years, ndvi_h, marker='o', linestyle='solid', label=f'{sample_id} High Tide')

    if tide_option in ["low", "both"]:
        ndvi_l = [sample_df[f'NDVI_{year}L'].values[0] for year in years]
        plt.plot(years, ndvi_l, marker='s', linestyle='dashed', label=f'{sample_id} Low Tide')

plt.xlabel("Year")
plt.ylabel("NDVI Value")
plt.title(f"NDVI Trends for Multiple Points (First {num_points} OBJECTIDs)")
plt.legend(loc='best', fontsize='small')
plt.grid()

# save to a png
#plt.savefig("ndvi_trends_multiple_points.png")
plt.show()

# import pandas as pd


# this file has 3 "files" in it, as i was adding wave data to the csv file in iterations one after the other.

# GenAI was used to debug errors and suggest changes to my code when needed

# creating yearly averages of below data

# wave_freq_file = "D:/FYP/2013_2023 Data/Wave Frequency.csv"     # Hourly wave frequency data
# sig_wave_file = "D:/FYP/2013_2023 Data/Significant Wave Height per Day.csv"  # Hourly significant wave height data
# output_file = "D:/FYP/Data/yearly_averages.csv"  # Output CSV file


# wave_df = pd.read_csv(wave_freq_file)
#

# wave_df['time'] = pd.to_datetime(wave_df['time'])
# wave_df['year'] = wave_df['time'].dt.year
#
# # Group by year, longitude, and latitude, then compute the average
# agg_wave = wave_df.groupby(['year', 'longitude', 'latitude'])['WaveFrequency'].mean().reset_index()
#
#
# pivot_wave = agg_wave.pivot_table(index=['longitude', 'latitude'],
#                                   columns='year',
#                                   values='WaveFrequency')
# # Rename pivoted columns, e.g. 2013 -> "WaveFrequency2013"
# pivot_wave.columns = [f"WaveFrequency{int(col)}" for col in pivot_wave.columns]
# pivot_wave = pivot_wave.reset_index()
#
# print("Yearly Average Wave Frequency:")
# print(pivot_wave)
# print("\n" + "="*80 + "\n")
#

# sig_wave_df = pd.read_csv(sig_wave_file)
#
# # Convert the time column to datetime and extract the year
# sig_wave_df['time'] = pd.to_datetime(sig_wave_df['time'])
# sig_wave_df['year'] = sig_wave_df['time'].dt.year
#
# # Group by year, longitude, and latitude, then compute the average
# agg_sig = sig_wave_df.groupby(['year', 'longitude', 'latitude'])['SignificantWaveHeight'].mean().reset_index()

# pivot_sig = agg_sig.pivot_table(index=['longitude', 'latitude'],
#                                 columns='year',
#                                 values='SignificantWaveHeight')
# # Rename pivoted columns, e.g. 2013 -> "SigWaveHeight2013"
# pivot_sig.columns = [f"SigWaveHeight{int(col)}" for col in pivot_sig.columns]
# pivot_sig = pivot_sig.reset_index()
#
# print("Yearly Average Significant Wave Height:")
# print(pivot_sig)
# print("\n" + "="*80 + "\n")


# merged_pivot = pd.merge(pivot_wave, pivot_sig, on=['longitude', 'latitude'], how='outer')
#
# print("Merged Yearly Averages (Wave Frequency & Significant Wave Height):")
# print(merged_pivot)

# merged_pivot.to_csv(output_file, index=False)



#_---------------------------------------------------------------------------------

# adding yearly averages to coordinate points (sign wave height, period etc.)

# import pandas as pd
#
# # File paths (adjust as needed)
# base_file = "D:/FYP/Data/updated_data.csv"                     # Base updated data file
# yearly_avg_file = "D:/FYP/Data/yearly_averages.csv"             # CSV with yearly averages (one row of averages)
# output_file = "D:/FYP/Data/updated_data_with_yearly_averages.csv"
#
# # Load the base updated data
# base_df = pd.read_csv(base_file)
#
# # Load the yearly averages data
# yearly_df = pd.read_csv(yearly_avg_file)

# if len(yearly_df) > 1:
#     avg_values = yearly_df.mean()
# else:
#     avg_values = yearly_df.iloc[0]

# for col, val in avg_values.items():
#     base_df[col] = val

# base_df.to_csv(output_file, index=False)


##-------------------------------------------------------------------------------------------------------


# adding tidal heights
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# File paths (adjust as needed)
base_file = "D:/FYP/Data/updated_data_with_yearly_averages.csv"  # Base file with wave averages (and other data)
tide_file = "D:/FYP/Data/yearly_tide_averages.csv"               # File with tide averages (with station_id, longitude, latitude, and tide columns)
output_file = "D:/FYP/Data/updated_data_with_tide_averages.csv"    # Output merged file

base_df = pd.read_csv(base_file)
tide_df = pd.read_csv(tide_file)

base_coords = base_df[['Longitude', 'Latitude']].values
tide_coords = tide_df[['longitude', 'latitude']].values

# Create and fit the Nearest Neighbors model on the tide coordinates.
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tide_coords)
distances, indices = nbrs.kneighbors(base_coords)

nearest_indices = indices.flatten()

tide_value_columns = [col for col in tide_df.columns if col not in ['station_id', 'longitude', 'latitude']]

#points given tide values of the sensor they are closest to
nearest_tide_df = tide_df.iloc[nearest_indices].reset_index(drop=True)

for col in tide_value_columns:
    base_df[col] = nearest_tide_df[col].values

base_df.to_csv(output_file, index=False)

print("Merged DataFrame preview (first 10 rows):")
print(base_df.head(10))
print(f"\nMerged file with tide averages saved to: {output_file}")




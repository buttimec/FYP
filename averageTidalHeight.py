
import pandas as pd

# Calculating average tidal height for eacch year for each sensor csv file

# List of tide-level CSV files
tide_files = [
    "D:/FYP/2013_2023 Data/Dublin Port Tide Height.csv",
    "D:/FYP/2013_2023 Data/Howth 1 Tide Height.csv",
    "D:/FYP/2013_2023 Data/Howth 2 Tide Height.csv",
    "D:/FYP/2013_2023 Data/Skerries Tide Height.csv",
]

output_file = r"D:\FYP\Data\yearly_tide_averages.csv"

df_list = []
for file_path in tide_files:
    #skip the second line (line index=1) which had "UTC degrees_east degrees_north metres" from the sensor
    temp_df = pd.read_csv(file_path, skiprows=[1])
    df_list.append(temp_df)

tide_df = pd.concat(df_list, ignore_index=True)

tide_df['time'] = pd.to_datetime(tide_df['time'])


tide_df['date'] = tide_df['time'].dt.date  # day
tide_df['year'] = tide_df['time'].dt.year  # year

#GenAI assisted method
daily_extremes = tide_df.groupby(
    ['station_id', 'longitude', 'latitude', 'date'],
    as_index=False
).agg(
    daily_min=('Water_Level_OD_Malin', 'min'),
    daily_max=('Water_Level_OD_Malin', 'max')
)


date_year_map = tide_df[['date', 'year']].drop_duplicates()

daily_extremes = daily_extremes.merge(date_year_map, on='date', how='left')

# grouped by station, coordinates, and year to get the average daily_min/daily_max
yearly_averages = daily_extremes.groupby(
    ['station_id', 'longitude', 'latitude', 'year'],
    as_index=False
).agg(
    avg_low_tide=('daily_min', 'mean'),
    avg_high_tide=('daily_max', 'mean')
)

pivot_df = yearly_averages.pivot_table(
    index=['station_id', 'longitude', 'latitude'],
    columns='year',
    values=['avg_low_tide', 'avg_high_tide']
)

new_cols = []
for col_tuple in pivot_df.columns:
    measure = col_tuple[0]
    year_val = col_tuple[1]
    if measure == 'avg_low_tide':
        new_cols.append(f'AvgLowTide{year_val}')
    else:
        new_cols.append(f'AvgHighTide{year_val}')

pivot_df.columns = new_cols
pivot_df = pivot_df.reset_index()


pivot_df.to_csv(output_file, index=False)

print("Yearly Tide Averages (first 10 rows):")
print(pivot_df.head(10))
print(f"\nSaved yearly tide averages to: {output_file}")

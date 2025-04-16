import pandas as pd
import os

# Adding the wave data to the sediment data csv file (for each coordinate point)

# Load the data
file_path = 'D:/FYP/Data/normalized_ndvi_data_with_sediment_with_wave_averages.xlsx'
df = pd.read_excel(file_path)

# Define the wave, period, power, and temperature values for each year (these averages were
# calculated using excel column inspection) - GenAI formatted to save time
wave_height = {
    2013: 1.372040, 2014: 1.537517, 2015: 1.209837, 2016: 1.078445, 2017: 1.101889,
    2018: 1.130265, 2019: 1.214239, 2020: 1.251988, 2021: 1.057941, 2022: 1.129306, 2023: 1.196401
}

wave_period = {
    2013: 4.392469, 2014: 4.507359, 2015: 4.263379, 2016: 4.164052, 2017: 4.135845,
    2018: 4.301591, 2019: 4.293652, 2020: 4.271574, 2021: 4.066183, 2022: 4.129948, 2023: 4.196574
}

wave_power = {
    2013: 38767.248787, 2014: 108543.527287, 2015: 85702.127959, 2016: 56299.811236, 2017: 55961.850475,
    2018: 43555.533796, 2019: 10445.982850, 2020: 19496.534047, 2021: 66545.687402, 2022: 66517.005470, 2023: 70361.272751
}

sea_temp = {
    2013: 9.655096, 2014: 13.151227, 2015: 11.201816, 2016: 9.501364, 2017: 11.906154,
    2018: 13.193411, 2019: 13.923848, 2020: 11.685483, 2021: 12.043900, 2022: 12.544681, 2023: 12.562113
}
#make new columns in csv and add the data in
for year in range(2013, 2024):
    df[f'WaveHeight{year}'] = wave_height[year]
    df[f'WavePeriod{year}'] = wave_period[year]
    df[f'WavePower{year}'] = wave_power[year]
    df[f'SeaTemp{year}'] = sea_temp[year]

output_file_path = os.path.join(os.path.dirname(file_path), 'updated_data.xlsx')
df.to_excel(output_file_path, index=False)

print(f"The new columns have been added and the data has been saved to {output_file_path}.")

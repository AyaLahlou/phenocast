import argparse
import pandas as pd
import numpy as np
import math

def calculate_photoperiod_vectorized(latitude, day_of_year):
    """
    Calculate photoperiod (duration of daylight) based on latitude and day of the year.
    
    Parameters:
    - latitude: Latitude in degrees (positive for northern hemisphere, negative for southern hemisphere).
    - day_of_year: Day of the year (1-365 or 1-366 for leap years).
    
    Returns:
    - photoperiod: Duration of daylight in hours.
    """
    # Convert latitude to radians
    lat_rad = np.radians(latitude)
    
    # Calculate solar declination
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # Calculate hour angle at sunrise and sunset
    hour_angle = np.arccos(-np.tan(lat_rad) * np.tan(np.radians(declination)))
    
    # Calculate photoperiod
    photoperiod = (2 * np.degrees(hour_angle)) / 15  # Convert angle to hours
    
    return photoperiod

def merge_yearly_data(parquet_file_path):
    
    filename_1= 'full_'
    filename_2= '_daily_NoNAN.parquet'
    # Read the Parquet file into a DataFrame using pyarrow
    full_df=pd.read_parquet(parquet_file_path+'/'+filename_1+str(1982)+filename_2, engine='pyarrow')
    for i in range (1983,2022):
        print('merging', i)
        full_df = pd.concat([full_df, pd.read_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, engine='pyarrow')], ignore_index=True)
    full_df.to_parquet(parquet_file_path+'_1982_2021.parquet', compression='snappy', engine='pyarrow')

def main(parquet_file_path):
    
    #merge files 
    print('step 1 : merging files')
    merge_yearly_data(parquet_file_path)
    print('Merge Completed! Results saved in'+ parquet_file_path+'_1982_2021.parquet')
    
    # Read data from parquet file
    data_df = pd.read_parquet(parquet_file_path+'_1982_2021.parquet')

    # Read soil data
    soil_df = pd.read_parquet('/Users/ayalahlou/Desktop/All Data/soil_type_data/GLDASp5_soiltexture_025d_processed.parquet')
    
    # Merge dataframes
    print('step 2 : adding soil type')
    data_df2 = pd.merge(data_df, soil_df, on=['latitude', 'longitude'], how='left')
    print('step 2 : adding soil type')

    # Convert 'time' column to datetime and calculate day of year (DOY)
    data_df2['time'] = pd.to_datetime(data_df2['time'])
    data_df2['DOY'] = data_df2['time'].dt.dayofyear
    
    # Calculate photoperiod
    data_df2['photoperiod'] = calculate_photoperiod_vectorized(data_df2['latitude'].values, data_df2['DOY'].values)
    
    # Drop 'DOY' column
    data_df2 = data_df2.drop(columns=['DOY'])
    
    # Write back to parquet file
    data_df2.to_parquet(parquet_file_path+'_1982_2021.parquet')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate photoperiod and update parquet file.")
    parser.add_argument("data_path", type=str, help="Path to the data file.")
    args = parser.parse_args()
    
    # Call main function with provided data_path
    main(args.data_path)
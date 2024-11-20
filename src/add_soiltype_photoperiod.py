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
    value = -np.tan(lat_rad) * np.tan(np.radians(declination))
    
    # Ensure the value is within the valid range for arccos
    if value < -1:
        print('value < -1 for latidue:', latitude, 'day of year:', day_of_year)
        value = -1
    elif value > 1:
        print('value > 1 for latidue:', latitude, 'day of year:', day_of_year)
        value = 1

    try:
        hour_angle = np.arccos(value)
    except ValueError as e:
        print(f"lat_rad: {lat_rad}, declination: {declination}")
    # Calculate photoperiod
    photoperiod = (2 * np.degrees(hour_angle)) / 15  # Convert angle to hours
    
    return photoperiod

def calculate_photoperiod_vectorized(latitudes, doys):
    lat_rad = np.radians(latitudes)
    declination = 23.44 * np.sin(np.radians((360 / 365) * (doys - 81)))
    value = -np.tan(lat_rad) * np.tan(np.radians(declination))
    
    # Ensure the value is within the valid range for arccos
    value = np.clip(value, -1, 1)
    
    hour_angle = np.arccos(value)
    photoperiod = 2 * (hour_angle * 24) / (2 * np.pi)
    
    return photoperiod

def merge_yearly_data(parquet_file_path):
    
    filename_1= 'full_'
    filename_2= '_daily_NoNAN.parquet'
    # Read the Parquet file into a DataFrame using pyarrow
    full_df=pd.read_parquet(parquet_file_path+'/'+filename_1+str(1982)+filename_2, engine='pyarrow')
    for i in range (1983,2022): #should be 1983, 2022
        print('merging', i)
        full_df = pd.concat([full_df, pd.read_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, engine='pyarrow')], ignore_index=True)
    full_df.to_parquet(parquet_file_path+'_1982_2021.parquet', compression='snappy', engine='pyarrow')

def main(parquet_file_path):
        
    # Read data from parquet file
    filename_1= 'full_'
    filename_2= '_daily_NoNAN.parquet'
    # Read the Parquet file into a DataFrame using pyarrow
    # Read soil data
    print('step 1 : adding soil type')
    soil_df = pd.read_parquet('/Users/ayalahlou/Desktop/All Data/soil_type_data/GLDASp5_soiltexture_025d_processed.parquet')
    for i in range (1982,2022):
        #read yearly era5 csif data 
        data_df = pd.read_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, engine='pyarrow')
        #merge soil data with era5 data
        if 'soil_x'  in data_df.columns or 'soil_y' in data_df.columns:
            try:
                data_df = data_df.drop(columns=['soil_x'])
            except:
                print('soil_x not found')
            try:
                data_df = data_df.drop(columns=['soil_y'])
            except:
                print('soil_y not found')
            
        data_df2 = pd.merge(data_df, soil_df, on=['latitude', 'longitude'], how='left')
        data_df2.to_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, compression='snappy', engine='pyarrow')
        print('soil type saved to', parquet_file_path+'/'+filename_1+str(i)+filename_2)

    print('step 2 : adding photoperiod')
    # add photoperiod
    for i in range (1982,2022):
        #read yearly era5 csif data 
        data_df = pd.read_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, engine='pyarrow')
        
        if 'photoperiod'  in data_df.columns :
            data_df = data_df.drop(columns=['photoperiod'])
        # Convert 'time' column to datetime and calculate day of year (DOY)
        data_df['time'] = pd.to_datetime(data_df['time'])
        data_df['DOY'] = data_df['time'].dt.dayofyear
        data_df['photoperiod'] = calculate_photoperiod_vectorized(data_df['latitude'].values, data_df['DOY'].values)

        # Drop 'DOY' column
        data_df = data_df.drop(columns=['DOY'])
        
        # Write back to parquet file
        data_df.to_parquet(parquet_file_path+'/'+filename_1+str(i)+filename_2, compression='snappy', engine='pyarrow')
        print('photoperiod saved to', parquet_file_path+'/'+filename_1+str(i)+filename_2)



        
    print('step 3 : merging files & save to '+parquet_file_path+'_1982_2021.parquet')
    merge_yearly_data(parquet_file_path)
    print('Merge Completed! Results saved in'+ parquet_file_path+'_1982_2021.parquet')


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="add soiltype and photoperiod and merge yearly parquet file.")
    parser.add_argument("data_path", type=str, help="Path to the data file.")
    args = parser.parse_args()
    
    # Call main function with provided data_path
    main(args.data_path)

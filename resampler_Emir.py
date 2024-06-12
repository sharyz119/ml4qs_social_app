import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns 

def resampler(some_df, rate='50L', method="downscale"):
    pd.options.mode.chained_assignment = None #remove the annoying warning
    
    # Create an empty DataFrame to store resampled data
    resampled_dfs = []

    # Group the DataFrame by 'source_name'
    grouped = some_df.groupby('source_name')
    
    # Iterate over each group
    for name, group in grouped:
        # Copy the group to avoid modifying the original DataFrame
        group_df = group.copy()
        group_df['time'] = pd.to_timedelta(group_df['time'], unit='s')
        
        # Set "time" as the index
        group_df.set_index('time', inplace=True)

        if method=="downscale" or method=="d":
            # Resample the data to the specified frequency, and choose the extreme value always
            resampled_group = group_df.drop(columns=['source_name']).resample(rate).apply(lambda x: max(x, key=abs)).interpolate()
        elif method == "upscale" or method=='u':
            resampled_group = group_df.drop(columns=['source_name']).resample(rate).ffill()
        #put back the source_name to the dataframe
        resampled_group['source_name'] = name
        
        # Reset the index to have "time" as a column again, and convert it back to readable seconds
        resampled_group.reset_index(inplace=True)
        resampled_group['time'] = resampled_group['time'].dt.total_seconds()

        # Store the resampled DataFrame for this category
        resampled_dfs.append(resampled_group)

    # Concatenate all resampled DataFrames into a single DataFrame
    resampled_df = pd.concat(resampled_dfs)
    
    # Return the resampled DataFrame
    return resampled_df


# my_path = "C:/Users/ameer/Desktop/AI/ML4QS/ml4qs_social_app/Emir_Datasets"

# accel = pd.read_parquet(my_path+"/"+"all_acceleration.parquet.gzip")
# lin_accel = pd.read_parquet(my_path+"/"+"all_lin_acceleration.parquet.gzip")
# gyro = pd.read_parquet(my_path+"/"+"all_gyroscope.parquet.gzip")
# baro = pd.read_parquet(my_path+"/"+"all_barometer.parquet.gzip")

# accel_resampled = resampler(accel)
# lin_accel_resampled = resampler(lin_accel)
# gyro_resampled = resampler(gyro)
# baro_resampled = resampler(baro, method="u") #since baro data are have rough granulaity in the raw data (~1 measurement/sec), we expand each measurement to duplicate values for each 0.05 sec (ie. downscaling)

# accel_resampled.to_parquet(my_path+"/"+"accel_resampled.parquet.gzip",
#               compression='gzip')
# lin_accel_resampled.to_parquet(my_path+"/"+"lin_accel_resampled.parquet.gzip",
#               compression='gzip')
# gyro_resampled.to_parquet(my_path+"/"+"gyro_resampled.parquet.gzip",
#               compression='gzip')
# baro_resampled.to_parquet(my_path+"/"+"baro_resampled.parquet.gzip",
#               compression='gzip')
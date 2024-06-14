import pandas as pd
from feature_engineering_Emir import fourier_transformer

def resampler(some_df, rate='50L', method="downscale", expand_features = True, acceleretor_fourier = False):
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
        resampled_group = pd.DataFrame()
        if method=="downscale" or method=="d":
            # Resample the data to the specified frequency, and choose the extreme value always
           if expand_features:
            my_col_names = group_df.drop(columns=['source_name']).columns
            resampling = group_df.drop(columns=['source_name']).resample(rate)
            resampled_group[[x+"_max" for x in my_col_names]] = resampling.max().interpolate()
            resampled_group[[x+"_min" for x in my_col_names]] = resampling.min().interpolate()
            resampled_group[[x+"_sd" for x in my_col_names]] = resampling.std().interpolate()
            resampled_group[[x+"_mean" for x in my_col_names]] = resampling.mean().interpolate()
            if acceleretor_fourier:
                tmp = resampling.apply(lambda x : fourier_transformer(x, sampling_interval=float(''.join(filter(str.isdigit, rate)))*0.001))
                resampled_group[[x+"_dominant_freq" for x in my_col_names]] = tmp.applymap(lambda x: x[0])
                resampled_group[[x+"_avg_weigh_freq" for x in my_col_names]] = tmp.applymap(lambda x: x[1])
           else:
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
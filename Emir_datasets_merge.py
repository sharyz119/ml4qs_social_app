import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns 
from feature_engineering_Emir import zscore_standardize_columns

my_path = "C:/Users/ameer/Desktop/AI/ML4QS/Our data"

onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
onlyfolds = [f for f in listdir(my_path) if not isfile(join(my_path, f))]
all_files_and_folds = listdir(my_path)

def renaming_func(some_df):
    col_names = some_df.columns
    for nms in col_names:
        if "Time" in nms:
            some_df = some_df.rename(columns={nms:"time"})
        elif "X " in nms or " x " in nms:
            some_df = some_df.rename(columns={nms:"x"})
        elif "Y " in nms or " y " in nms:
            some_df = some_df.rename(columns={nms:"y"})
        elif "Z " in nms or " z " in nms:
            some_df = some_df.rename(columns={nms:"z"})
        elif "Pressure" in nms:
            some_df = some_df.rename(columns={nms:"pressure"})
        # else:
        #     some_df = some_df.rename(columns={nms:"source_name"})
    return some_df

acel = "Accelerometer.csv" 
linear_acel = "Linear Accelerometer.csv"
gyro = "Gyroscope.csv"
baro = "Barometer.csv"

file_names = [acel, linear_acel, gyro, baro]
file_names2 = ["Accelerometer.csv", "Linear Acceleration.csv", "Gyroscope.csv", "Pressure.csv"]

accel_df1 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
lin_accel_df1 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
gyro_df1 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
baro_df1 = pd.DataFrame(columns=["time", "x", "source_name"],dtype=object)

df_list = [accel_df1,lin_accel_df1,gyro_df1,baro_df1]

for i in onlyfolds: #iterate through folders of social media measurements eg. "Red4 2024-06-06 20-02-55"
    for count,j in enumerate(file_names): #iterate through files in the folder
        my_df = pd.read_csv(my_path+"/"+str(i)+"/"+j) #read the tables in it
        my_df["source_name"] = str(i).split()[0] #add the name of Social Media source to a column in the dataset
        my_df = renaming_func(my_df) #simplify column names
        df_list[count] = pd.concat([df_list[count], my_df], ignore_index=True) #add the simplified df to the new dfs list
        
accel_df1 = df_list[0]
lin_accel_df1 = df_list[1]
gyro_df1 = df_list[2]
baro_df1 = df_list[3]

accel_df2 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
lin_accel_df2 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
gyro_df2 = pd.DataFrame(columns=["time", "x", "y", "z","source_name"],dtype=object)
baro_df2 = pd.DataFrame(columns=["time", "x", "source_name"],dtype=object)

df_list2 = [accel_df2,lin_accel_df2,gyro_df2,baro_df2]
deicde_list = lambda x: file_names2 if ("FB" in x) else file_names


for i in onlyfiles: #names of zip files
        for count, j in enumerate(deicde_list(i)): #names of csv files
                with zipfile.ZipFile(my_path+"/"+str(i), 'r') as zip_ref: #unzip the zip file
                        with zip_ref.open(j) as file: #take out the csv file
                            my_df = pd.read_csv(file) #have the csv in a df
                            my_df["source_name"] = str(i).split()[0] #add the name of the source 
                            my_df = renaming_func(my_df) #change the names of the columns
                            df_list2[count] = pd.concat([df_list2[count], my_df], ignore_index=True) #take out the df and append the data to it

accel_df2 = df_list2[0]
lin_accel_df2 = df_list2[1]
gyro_df2 = df_list2[2]
baro_df2 = df_list2[3]
baro_df2['x'] = baro_df2['x'].fillna(baro_df2['pressure'])
baro_df2 = baro_df2.drop(columns=['pressure'])

accel_df1 = pd.concat([accel_df1,accel_df2], ignore_index=True)
lin_accel_df1 = pd.concat([lin_accel_df1,lin_accel_df2], ignore_index=True)
gyro_df1 = pd.concat([gyro_df1,gyro_df2], ignore_index=True)
baro_df1 = pd.concat([baro_df1,baro_df2], ignore_index=True)

# accel_df1.to_parquet(my_path+"/"+"all_acceleration.parquet.gzip", index = False,compression='gzip')
# lin_accel_df1.to_parquet(my_path+"/"+"all_lin_acceleration.parquet.gzip", index = False,compression='gzip')
# gyro_df1.to_parquet(my_path+"/"+"all_gyroscope.parquet.gzip", index = False,compression='gzip')
# baro_df1.to_parquet(my_path+"/"+"all_barometer.parquet.gzip", index = False,compression='gzip')

# accel = pd.read_parquet(my_path+"/all_acceleration.parquet.gzip")
# lin_accel = pd.read_parquet(my_path+"/all_lin_acceleration.parquet.gzip")
# baro = pd.read_parquet(my_path+"/all_barometer.parquet.gzip")
# gyro = pd.read_parquet(my_path+"/all_gyroscope.parquet.gzip")

accel_resampled = accel_resampled.rename(columns=lambda x: "accel_" + x)
accel_resampled = accel_resampled.rename(columns= {"accel_time":"time", "accel_source_name":"source_name"})
lin_accel_resampled = lin_accel_resampled.rename(columns=lambda x: "linaccel_" + x)
lin_accel_resampled = lin_accel_resampled.rename(columns= {"linaccel_time":"time", "linaccel_source_name":"source_name"})
gyro_resampled = gyro_resampled.rename(columns=lambda x: "gyro_" + x)
gyro_resampled = gyro_resampled.rename(columns= {"gyro_time":"time", "gyro_source_name":"source_name"})
baro_resampled = baro_resampled.rename(columns=lambda x: "baro_" + x)
baro_resampled = baro_resampled.rename(columns= {"baro_time":"time", "baro_source_name":"source_name"})

def indexing_func(df):
    indecies = []
    grouped = df.groupby('source_name')
    for name, group in grouped:
        indecies.extend(np.arange(0,len(group)))
    return indecies

accel_resampled["index"] = indexing_func(accel_resampled)
lin_accel_resampled["index"] = indexing_func(lin_accel_resampled)
gyro_resampled["index"] = indexing_func(gyro_resampled)
baro_resampled["index"] = indexing_func(baro_resampled)

merged_df = pd.merge(accel_resampled, lin_accel_resampled, on=['index', 'source_name'], suffixes=('_accel', '_linaccel'))
merged_df = pd.merge(merged_df, gyro_resampled, on=['index', 'source_name'], suffixes=('_mergedpre', '_gyro'))
merged_df = pd.merge(merged_df, baro_resampled, on=['index', 'source_name'], suffixes=('_mergedpost', '_baro'))

#For some reason, the gyroscope reddit2 dataset is not saving values, this saves the values manually to the dataset

NA_cols = ["gyro_zscore_x_max","gyro_zscore_y_max", "gyro_zscore_z_max","gyro_zscore_x_min",
           "gyro_zscore_y_min", "gyro_zscore_z_min", "gyro_zscore_x_sd", "gyro_zscore_y_sd",
           "gyro_zscore_z_sd", "gyro_zscore_x_mean", "gyro_zscore_y_mean", "gyro_zscore_z_mean"]

new_df_names = [s.replace("gyro_", "") for s in NA_cols]
modified_strings = [s.replace("gyro_zscore_", "") for s in NA_cols]
gyro_zscores = zscore_standardize_columns(gyro_resampled[gyro_resampled["source_name"]=="Reddit2"][["source_name","time"]+modified_strings][0:5955]).drop(columns=["source_name","time"])
gyro_zscores.rename(columns=dict(zip(new_df_names, NA_cols)), inplace=True)
gyro_zscores.index = merged_df[merged_df["source_name"]=="Reddit2"][NA_cols].index
merged_df.loc[merged_df["source_name"]=="Reddit2", NA_cols] = gyro_zscores

# my_path = "C:/Users/ameer/Desktop/AI/ML4QS/ml4qs_social_app/Emir_Datasets"

# merged_df.to_parquet(my_path+"/full_dataset(some NAs).parquet.gzip", compression='gzip')

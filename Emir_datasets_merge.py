import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns 

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

for i in onlyfolds:
    for count,j in enumerate(file_names):
        my_df = pd.read_csv(my_path+"/"+str(i)+"/"+j)
        my_df["source_name"] = str(i).split()[0]
        my_df = renaming_func(my_df)
        df_list[count] = df_list[count].append(my_df, ignore_index=True)
        #accel_df1 = accel_df1.append(my_df, ignore_index=True)
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
                            df_list2[count] = df_list2[count].append(my_df, ignore_index=True) #take out the df and append the data to it

accel_df2 = df_list2[0]
lin_accel_df2 = df_list2[1]
gyro_df2 = df_list2[2]
baro_df2 = df_list2[3]

accel_df1 = accel_df1.append(accel_df2, ignore_index=True)
lin_accel_df1 = lin_accel_df1.append(lin_accel_df2, ignore_index=True)
gyro_df1 = gyro_df1.append(gyro_df2, ignore_index=True)
baro_df1 = baro_df1.append(baro_df2, ignore_index=True)

# accel_df1.to_csv(my_path+"/"+"all_acceleration.csv", index = False)
# lin_accel_df1.to_csv(my_path+"/"+"all_lin_acceleration.csv", index = False)
# gyro_df1.to_csv(my_path+"/"+"all_gyroscope.csv", index = False)
# baro_df1.to_csv(my_path+"/"+"all_barometer.csv", index = False)

# accel = pd.read_csv(my_path+"/"+"all_acceleration.csv")
# lin_accel = pd.read_csv(my_path+"/"+"all_lin_acceleration.csv")
# gyro = pd.read_csv(my_path+"/"+"all_gyroscope.csv")
# baro = pd.read_csv(my_path+"/"+"all_barometer.csv")
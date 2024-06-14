import numpy as np
import pandas as pd
from scipy.stats import zscore


#add an inclination function
def inclination_extractor(some_df):
    return np.degrees(np.arctan2(some_df["y"],some_df["z"]))

#add z scores
def zscore_standardize_columns(some_df):

    results_df = pd.DataFrame()
    grouped = some_df.groupby('source_name')

    for name, group in grouped:
        temp_df = pd.DataFrame()
        group_df = group.copy()
        group_df = group_df.drop(columns=['time'])
        s = group_df.select_dtypes("number").columns
        temp_df[["zscore_"+x for x in group_df.loc[:,s].columns]] = group_df.loc[:,s].apply(zscore)
        temp_df[['time','source_name']] = group[['time','source_name']]
        results_df = pd.concat([results_df, temp_df])
    return results_df
           
def fourier_transformer(some_column, sampling_interval=0.05):
    fft_column = np.fft.fft(some_column)
    frequencies = np.fft.fftfreq(len(fft_column), d=sampling_interval)
    power_spectrum = np.abs(fft_column)
    weighted_sum = np.sum(frequencies * power_spectrum)
    total_magnitude = np.sum(power_spectrum)
    frequency_weighted_avg = weighted_sum / total_magnitude
    dominant_frequency = frequencies[np.argmax(power_spectrum)]
    return dominant_frequency, frequency_weighted_avg

#accel["inclination"] = inclination_extractor(accel)

# new_baro = zscore_standardize_columns(baro)
# new_accel = zscore_standardize_columns(accel)
# new_lin_accel = zscore_standardize_columns(lin_accel)
# new_gyro = zscore_standardize_columns(gyro)

# all_baro = pd.merge(baro, new_baro, on = ['time', 'source_name'])
# all_gyro = pd.merge(gyro, new_gyro, on = ['time', 'source_name'])
# all_accel = pd.merge(accel, new_accel, on = ['time', 'source_name'])
# all_lin_accel = pd.merge(lin_accel, new_lin_accel, on = ['time', 'source_name'])

# my_path = 'C:/Users/ameer/Desktop/AI/ML4QS/ml4qs_social_app/Emir_Datasets'

# all_accel.to_parquet(my_path+"/"+"all_acceleration.parquet.gzip", index = False,compression='gzip')
# all_lin_accel.to_parquet(my_path+"/"+"all_lin_acceleration.parquet.gzip", index = False,compression='gzip')
# all_gyro.to_parquet(my_path+"/"+"all_gyroscope.parquet.gzip", index = False,compression='gzip')
# all_baro.to_parquet(my_path+"/"+"all_barometer.parquet.gzip", index = False,compression='gzip')
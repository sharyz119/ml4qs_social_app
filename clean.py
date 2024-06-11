import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.special import erfinv

# Read the combined CSV file
df = pd.read_csv('X3/combined_resampled_data.csv')

# Calculate the magnitude of the horizontal component
df['H'] = np.sqrt(df['Accel_X (m/s^2)']**2 + df['Accel_Y (m/s^2)']**2)

# Calculate the inclination angle in degrees
df['Inclination (degrees)'] = np.degrees(np.arctan(df['H'] / df['Accel_Z (m/s^2)']))


# Function to detect outliers using Chauvenet's criterion and replace them with the median
def replace_outliers_with_median(series):
    mean = series.mean()
    std = series.std()
    N = len(series)
    criterion = 1.0 / (2 * N)
    deviation = np.abs(series - mean) / std
    threshold = std * np.sqrt(2) * erfinv(1 - criterion)
    mask = deviation < threshold
    series_outliers_replaced = series.copy()
    series_outliers_replaced[~mask] = series.median()  # Replace outliers with median
    return series_outliers_replaced

# Apply Chauvenet's criterion and replace outliers for each numerical column
columns_to_check = [col for col in df.columns if col not in ['timestamp', 'app', 'Time (s)']]
df[columns_to_check] = df[columns_to_check].apply(replace_outliers_with_median)

# Function to apply lowpass filter
def lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Apply lowpass filter to each numerical column except 'timestamp' and 'Time (s)'
for col in columns_to_check:
    df[col] = lowpass_filter(df[col].values)


# Save the cleaned and processed dataframe to a new CSV file
df.to_csv('X3/data_X3.csv', index=False)

print("Data cleaned and saved to 'cleaned_combined_data_no_outliers_noise.csv'")

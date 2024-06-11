import pandas as pd

# Read the CSV files
accelerometer = pd.read_csv('X3/Accelerometer.csv')
barometer = pd.read_csv('X3/Barometer.csv')
gyroscope = pd.read_csv('X3/Gyroscope.csv')
linear_accelerometer = pd.read_csv('X3/Linear Accelerometer.csv')

# Step 1: Process Barometer - Retain whole numbers before the decimal point
barometer['timestamp'] = barometer['Time (s)'].apply(lambda x: int(float(x)))
barometer.set_index('timestamp', inplace=True)

# Step 2: Process Accelerometer, Gyroscope, and Linear Accelerometer
# Convert 'Time (s)' to timedelta for resampling
accelerometer['timestamp'] = pd.to_timedelta(accelerometer['Time (s)'], unit='s')
gyroscope['timestamp'] = pd.to_timedelta(gyroscope['Time (s)'], unit='s')
linear_accelerometer['timestamp'] = pd.to_timedelta(linear_accelerometer['Time (s)'], unit='s')

# Set the timestamp as the index for each dataframe
accelerometer.set_index('timestamp', inplace=True)
gyroscope.set_index('timestamp', inplace=True)
linear_accelerometer.set_index('timestamp', inplace=True)

# Resample using average to match the barometer's sampling rate (1Hz)
resample_frequency = '1S'

accelerometer_resampled = accelerometer.resample(resample_frequency).mean()
gyroscope_resampled = gyroscope.resample(resample_frequency).mean()
linear_accelerometer_resampled = linear_accelerometer.resample(resample_frequency).mean()

# Convert the index to integer seconds
accelerometer_resampled['timestamp'] = accelerometer_resampled.index.total_seconds().astype(int)
gyroscope_resampled['timestamp'] = gyroscope_resampled.index.total_seconds().astype(int)
linear_accelerometer_resampled['timestamp'] = linear_accelerometer_resampled.index.total_seconds().astype(int)

# Set the new timestamp as index
accelerometer_resampled.set_index('timestamp', inplace=True)
gyroscope_resampled.set_index('timestamp', inplace=True)
linear_accelerometer_resampled.set_index('timestamp', inplace=True)

# Step 3: Rename columns to avoid overlap
accelerometer_resampled.columns = ['Accel_' + col for col in accelerometer_resampled.columns]
gyroscope_resampled.columns = ['Gyro_' + col for col in gyroscope_resampled.columns]
linear_accelerometer_resampled.columns = ['LinAccel_' + col for col in linear_accelerometer_resampled.columns]

# Step 4: Combine all datasets
combined_df = accelerometer_resampled.join([barometer, gyroscope_resampled, linear_accelerometer_resampled], how='outer')

# Add a column 'app' and fill it with 'Red'
combined_df['app'] = 'X'

# Reset the index to include the timestamp in the CSV file
combined_df.reset_index(inplace=True)

# Save the combined dataframe to a CSV file
combined_df.to_csv('X3/combined_resampled_data.csv', index=False)

print("Data successfully merged and saved to 'combined_resampled_data_final.csv'")

import numpy as np

def inclination_extractor(some_df):
    return np.degrees(np.arctan2(some_df["y"],some_df["z"]))

#accel_resampled["inclination"] = inclination_extractor(accel_resampled)

# my_path = "C:/Users/ameer/Desktop/AI/ML4QS/ml4qs_social_app/Emir_Datasets"

# accel_resampled.to_parquet(my_path+"/"+"accel_resampled.parquet.gzip",
#               compression='gzip')
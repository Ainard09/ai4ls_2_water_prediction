import pandas as pd
import numpy as np
import os
from os import path
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import re
import traceback
from use_functions import find_temp_mps_in_radius



def csv_find_skiprows(filepath, lookup):
    skiprows = 0
    # first know how many rows to skip in pd.read_csv. TO do this will open the file and look for "Werte:"
    with open(filepath, encoding='unicode_escape') as f:
        for num, line in enumerate(f, 1):
            if lookup in line:
                skiprows = num
                # print("skiprows:", skiprows)
                break
                
    return skiprows

def process_region_gw_temp(region_dir, df_mps, location_id, radius=20000):

    val_col_name = "gw-level"
    folder_name = "Grundwasserstand-Monatsmittel"

    mnth_file_path = path.join(region_dir, "Grundwasserstand-Monatsmittel/" "Grundwasserstand-Monatsmittel-{0}.csv".format(location_id))
    
    print("> Processing {} - {}..".format(location_id, folder_name))
    
    # first know how many rows to skip in pd.read_csv. TO do this will open the file and look for "Werte:"
    lookup = "Werte:"
    skiprows = csv_find_skiprows(mnth_file_path, lookup)
    
    # load and process .csv file
    df = pd.read_csv(mnth_file_path, sep=";", header=None, skiprows=skiprows,
                            encoding='unicode_escape')
    
    # manipulate data splitting values into more possible regressors (e.g. month, year)
    df.columns = ["date", val_col_name, "empty"]
    del df["empty"] 
    
    df["date"] = pd.to_datetime(df["date"].str.strip(), format='%d.%m.%Y %H:%M:%S')
    df.set_index("date", inplace=True)
    datetime_seq = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')


    df2 = pd.DataFrame(datetime_seq, columns=["date"])
    
    # Use boolean indexing to assign values
    for date in datetime_seq:
        if date in df.index:
            value =  df.loc[date, val_col_name]
            df2.loc[df2['date'] == date, val_col_name] = value.replace(",",".") if "Lücke" not in value else value
        else:
            df2.loc[df2['date'] == date, val_col_name] = np.nan

    # Use regex to handle variations of "Lücke"
    regex_pattern = r'\bLücke\b' 

    # Replace variations of "Lücke" with NaN
    df2[val_col_name] = df2[val_col_name].astype(str).replace(regex_pattern, np.nan, regex=True)

    # Convert to float and handle NaN interpolation
    df2[val_col_name] = pd.to_numeric(df2[val_col_name], errors='coerce')
    df2[val_col_name] = df2[val_col_name].interpolate(method="linear")
    df2[val_col_name] = df2[val_col_name].round(2)
    
    # remove the last year 2022-01-01 on the date column
    last_date = df2['date'].max()
    if '2022-01-01' in str(last_date):
        df2 = df2.iloc[:-1]
    df = df2

    # # add more time series for test data
    last_date = df["date"].max()
    new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=26, freq='MS')

    # Create a DataFrame with the new dates
    new_data = pd.DataFrame({"date": new_dates})
    df = pd.concat([df, new_data], ignore_index=True)

    # add temperature
    region_temp_dir = region_dir + "/Grundwassertemperatur-Monatsmittel" 
    region_temp_filenames = os.listdir(region_temp_dir)
    
    temp_mps = []
    for file in region_temp_filenames:
        loc_id = file.split(".")[0].split("-")[-1]
        temp_mps.append(int(loc_id)) 

    if int(location_id) in temp_mps:
        temp_num_id = location_id
    else:
        # Set 20,000 radius around the measuring point area
        hzhnr01 = int(location_id)
        rds_temp_loc = find_temp_mps_in_radius(df_mps, temp_mps, hzhnr01, radius)
    
        # randomly selecta temperature area
        temp_num_id = str(np.random.choice(rds_temp_loc))

    mnth_temp_fpath = path.join(region_dir, "Grundwassertemperatur-Monatsmittel/" "Grundwassertemperatur-Monatsmittel-{0}.csv".format(temp_num_id))

    skiprows = csv_find_skiprows(mnth_temp_fpath, lookup)
    
    # load and process .csv file
    df_temp = pd.read_csv(mnth_temp_fpath, sep=";", header=None, skiprows=skiprows,
                            encoding='unicode_escape') 
    
    df_temp.columns = ["date", "temp", "empty"]
    del df_temp["empty"] 

    # remove rows with gaps ("Lücke")
    df_temp = df_temp[~df_temp["temp"].str.contains("Lücke")]
    
    df_temp["date"] = pd.to_datetime(df_temp["date"].str.strip(), format='%d.%m.%Y %H:%M:%S')

    # Create a dictionary for quick lookup
    temp_dict = df_temp.set_index("date")["temp"].to_dict()
    
    # Fill in temp values based on date
    df["temp"] = df["date"].map(temp_dict)
    
    # Convert temp column to numeric and handle commas
    df["temp"] = pd.to_numeric(df["temp"].str.replace(",", "."), errors='coerce')
    df["temp"] = df["temp"].interpolate(method="linear")
    df["temp"] = df["temp"].fillna(df["temp"].mean())
    df["temp"] = df["temp"].round(2)

    # Set the last 26 entries of the 'temp' column to NaN
    df.loc[df.index[-26:], 'temp'] = np.nan
        
    print("\t- Temperature Preprocess Done!")

    return df






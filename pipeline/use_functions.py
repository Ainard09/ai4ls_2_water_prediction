import pandas as pd
import numpy as np
import os
from os import path
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import re
import traceback



def get_num_from_str(string):
    matches = re.findall(r"\d+\.\d+", string)
    if len(matches):
        return matches[0]
    else:
        return np.nan

def get_mp_attrs(region_dir, mp_num):

    mp_base_data_path = path.join(region_dir, "Stammdaten", "Stammdaten-{0}.txt".format(mp_num))
    
    land_height = np.nan
    mp_height = np.nan
    bottom_line = np.nan
    t_measuring_depth = np.nan
    
    with open(mp_base_data_path, encoding='unicode_escape') as f:
        for num, line in enumerate(f, 1):
            if "Geländehöhe" in line:
                land_height = get_num_from_str(line)
                
            elif "Messpunkthöhe" in line:
                mp_height = get_num_from_str(line)
                
            elif "Sohllage" in line:
                bottom_line = get_num_from_str(line)
                
            elif "T-Messtiefe u.GOK" in line:
                t_measuring_depth = get_num_from_str(line)
                
                break # since it's always listed latest in the file
    
    mp = {
        "land_height": land_height,
        "mp_height": mp_height,
        "bottom_line": bottom_line,
        "t_measuring_depth": t_measuring_depth
    }
    
    return mp

def get_all_mps(region_dir):

    values = []

    # mps: measurment points 
    df_mps_path = path.join(region_dir, "messstellen_alle.csv")
    df_mps = pd.read_csv(df_mps_path, sep=";")
    
    # filter to typ == 'gw' then del typ col
    df_mps = df_mps.query("typ=='gw'")
    del df_mps["typ"]

    #df_mps["region"] = region

    # for every mp in the region get attributes
    # create attrs cols and init with np.nan
    mp_attrs = ["land_height", "mp_height", "bottom_line", "t_measuring_depth"]
    for mp_attr in mp_attrs:
        df_mps[mp_attr] = np.nan

    # fill with values when available
    for index, row in df_mps.iterrows():
        mp_num = row["hzbnr01"]
        mp_attr_vals = get_mp_attrs(region_dir, mp_num)
        for mp_attr in mp_attrs:
            df_mps.at[index, mp_attr] = float(mp_attr_vals[mp_attr])

    values.extend(df_mps.values.tolist())

    colnames = ["x", "y", "dbmsnr", "hzbnr01"] + mp_attrs
    df_mps_all = pd.DataFrame(values, columns = colnames)
    df_mps_all["x"] = df_mps_all["x"].str.replace(",", ".").astype(float)
    df_mps_all["y"] = df_mps_all["y"].str.replace(",", ".").astype(float)

    return df_mps_all

def find_temp_mps_in_radius(df_mps_all, temp_mps, hzbnr01, radius):

    # get mp coordiantes
    c_x = df_mps_all.query('hzbnr01==@hzbnr01').iloc[0]["x"]
    c_y = df_mps_all.query('hzbnr01==@hzbnr01').iloc[0]["y"]

    # find all mps in radius
    x_min = c_x - radius
    x_max = c_x + radius
    y_min = c_y - radius
    y_max = c_y + radius
    df_mps_rds = df_mps_all.query('(@x_min <= x <= @x_max) & (@y_min <= y <= @y_max)')
    
    # get the temp mps in the radius dataframe
    df_temp = df_mps_rds[df_mps_rds["hzbnr01"].isin(temp_mps)]
    
    return df_temp["hzbnr01"].tolist()


def smape(A, F):
    """ Define the function to return the SMAPE value """

    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return round(100 / len_ * np.nansum(tmp), 3)
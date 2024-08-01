import pandas as pd
import numpy as np
import os
from os import path
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import re
import traceback


# Define a function to map month to season for exogenous varibles
def get_season(month):
    if month in [3.0, 4.0, 5.0]:
        return 'spring'
    elif month in [6.0, 7.0, 8.0]:
        return 'summer'
    elif month in [9.0, 10.0, 11.0]:
        return 'Fall'
    else:  # months 12.0, 1.0, 2.0
        return 'winter'

# information from www.weatherspark.com site
def get_weather(month):
    if month in [1, 2, 12]:
        return "freezing"
    elif month in [3, 11]:
        return "cold"
    elif month in [4,10]:
        return "cool"
    elif month in [5, 6, 9]:
        return "comfortable"
    else: # 7 & 8
        return "warm"

def cyclical_encoded(data, cycle_length):
    """ function to capture pattern on calender features """

    sin = np.sin(2 * np.pi * data/cycle_length)
    cos = np.cos(2 * np.pi * data/cycle_length)
    result =  pd.DataFrame({
                  f"{data.name}_sin": sin,
                  f"{data.name}_cos": cos
              })

    return result


def add_exogenous_features(df, df_recursive_temp, mp_num):

    print("> Add Exogenous Features")
    
    val_col_name = "gw-level"

    # convert date to datetime column
    df.reset_index(inplace=True)
    df.columns = ["date","gw-level", "temp"]
    df["date"] = pd.to_datetime(df["date"])
    
    df_rescur_id = df_recursive_temp[["date", mp_num]]
    date_recur_results_dict = df_rescur_id.set_index("date")[mp_num].to_dict()

    # fill the temp nan with recursive results
    df.loc[df['temp'].isna(), 'temp'] = df.loc[df['temp'].isna(), 'date'].map(date_recur_results_dict)


    # add temp features
    df['temp_roll_mean_1_year'] = df['temp'].rolling(12, closed='left').mean()
    df['temp_roll_mean_2_year'] = df['temp'].rolling(24, closed='left').mean()
    df['temp_roll_max_1_year'] = df['temp'].rolling(12, closed='left').max()
    df['temp_roll_min_1_year'] = df['temp'].rolling(12, closed='left').min()

    
    # add calender features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # add season features
    df["season"] = df["date"].dt.month.apply(get_season)
    df["weather"] = df["date"].dt.month.apply(get_weather)
    df["season"] = df["season"].astype("category")
    df["weather"] = df["weather"].astype("category")
    
    # cyclic calender and seasonal features
    month_cyclic = cyclical_encoded(df["month"], cycle_length=24)
    quarter_cyclic = cyclical_encoded(df["quarter"], cycle_length=4)

    # merge the df to the cyclic the features
    df_exogenous_features = pd.concat([df,month_cyclic, quarter_cyclic], axis=1)

    # add intereaction between exogenous varibles
    transformer_poly = PolynomialFeatures(
    degree           = 2,
    interaction_only = True,
    include_bias     = False
    ).set_output(transform="pandas")

    # pick columns for exgennous varibles for intereactions
    copy_df = df_exogenous_features.copy()
    copy_df.drop(["season","weather","date",val_col_name], axis=1, inplace=True)
    poly_cols = copy_df.columns.tolist()

    poly_features = transformer_poly.fit_transform(df_exogenous_features[poly_cols].dropna())
    poly_features = poly_features.drop(columns=poly_cols)
    poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
    poly_features.columns = poly_features.columns.str.replace(" ", "_")
    df_exogenous_features = pd.concat([df_exogenous_features, poly_features], axis=1)

    # Set the last 26 entries of the 'temp' column to 0.0
    df_exogenous_features.loc[df_exogenous_features.index[-26:], 'gw-level'] = 0.0

    df_exogenous_features.dropna(inplace=True)
    df_exogenous_features['temp_roll_mean_1_year'] = df_exogenous_features['temp_roll_mean_1_year'].round(2)
    df_exogenous_features['temp_roll_mean_2_year'] = df_exogenous_features['temp_roll_mean_2_year'].round(2)

        
    print("\t- Exogenous Features Done!")

    return df_exogenous_features
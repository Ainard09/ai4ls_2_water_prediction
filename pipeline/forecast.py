import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import select_features
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import argparse
import re
import traceback
from use_functions import get_all_mps, smape
from preprocess_temp import process_region_gw_temp
from recursive_temp_pred import recursive_populate_template
from add_exogenous import add_exogenous_features


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('location_id', type=str, help='the geographical location ID')
    #parser.add_argument("region", type=str, help='The name of the region this location ID belongs')
    parser.add_argument("region_dir", type=str, help='The file directory path to the region dataset')
    
    args = parser.parse_args()
    return args


# one-hot encoding
categorical_features = ["weather", "season"]
transformer_exog = make_column_transformer(
    (
        OrdinalEncoder(
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1
        ),
        categorical_features
    ),
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")


# Regressor hyperparameters search space
def search_space(trial):

  # Lags grid
  lags_grid = tuple([12, 24, [1, 2, 3, 4, 7, 9, 24]])

  search_space  = {
      'max_iter'          : trial.suggest_int('max_iter', 400, 1200, step=100),
      'max_depth'         : trial.suggest_int('max_depth', 3, 10, step=1),
      'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 1),
      'min_samples_leaf'  : trial.suggest_int('min_samples_leaf', 1, 20, step=1),
      'l2_regularization' : trial.suggest_float('l2_regularization', 0, 1),
      'lags'              : trial.suggest_categorical('lags', lags_grid)
  }
  return search_space

def search_hyperparameters(data, end_train, end_valid, exog_features,transformer_exog):

  # instantiate a forcaster transformer with categorical features
  forecaster = ForecasterAutoreg(
  regressor = HistGradientBoostingRegressor(
                  categorical_features=categorical_features,
                  random_state=123
              ),
  lags = 24,
  transformer_exog = transformer_exog
  )

  # search for best parameters
  results_search, frozen_trial = bayesian_search_forecaster(
  forecaster         = forecaster,
  y                  = data.loc[:end_valid, 'gw-level'],
  exog               = data.loc[:end_valid, exog_features],
  search_space       = search_space,
  steps              = 30,
  refit              = False,
  metric             = 'mean_absolute_percentage_error',
  initial_train_size = len(data.loc[:end_train]),
  fixed_train_size   = False,
  n_trials           = 20,
  random_state       = 123,
  return_best        = True,
  n_jobs             = 'auto',
  verbose            = False,
  show_progress      = True
  )

  best_params = results_search['params'].iat[0]

  return best_params

def train_and_predict(data, best_params, actual_data, end_valid, end_train, valid_num, train_num, df_idx, exog_features, transformer_exog):

  # train for evaluation of the model
  forecaster = ForecasterAutoreg(
  regressor = HistGradientBoostingRegressor(**best_params,
                  categorical_features=categorical_features,
                  random_state=123
              ),
  lags = 24,
  transformer_exog = transformer_exog
  )

  # train the model the time series train and validation dataset
  forecaster.fit(
    y    = data.loc[:end_train, 'gw-level'],
    exog = data.loc[:end_train, exog_features]
  )

  # make predictions and evalute the model
  predictions = forecaster.predict(
      exog     = data.loc[df_idx[train_num+1]:, exog_features],
      steps    = 26
  )
  df_preds = pd.DataFrame(predictions)
  preds = df_preds["pred"].values
  # evaluate on symmetric mean absolute percentage error
  smape_value = smape(actual_data, preds)

  # train for future predictions
  forecaster = ForecasterAutoreg(
  regressor = HistGradientBoostingRegressor(**best_params,
                  categorical_features=categorical_features,
                  random_state=123
              ),
  lags = 24,
  transformer_exog = transformer_exog
  )
  # train the model the time series train and validation dataset
  forecaster.fit(
    y    = data.loc[:end_valid, 'gw-level'],
    exog = data.loc[:end_valid, exog_features]
  )

  # make predictions into the future
  predictions = forecaster.predict(
    exog     = data.loc[df_idx[valid_num+1]:, exog_features],
    steps    = 26
  )
  df_preds = pd.DataFrame(predictions)

  # free resources since it's going to run on iterations
  del forecaster

  return df_preds, smape_value


def populate_test_data(df_exog):

    print("> Start Training and Forecast")

    try:

        df_exog["season"] = df_exog["season"].astype("category")
        df_exog["weather"] = df_exog["weather"].astype("category")
        df_exog["date"] = pd.to_datetime(df_exog["date"])
        df_exog.set_index("date", inplace=True)
        df_exog.index = pd.date_range(start=df_exog.index.min(), end=df_exog.index.max(), freq='MS')

        # get the estimate end train and end validation dates
        data = df_exog.copy()
        exog_data = data.drop("gw-level", axis=1)
        exog_features = exog_data.columns
        df_idx = data.index
        train_num = int(len(data) * 0.8)
        valid_num = len(data.loc[:"2021-11-01"])
        end_train = df_idx[train_num]
        end_valid = df_idx[valid_num]
        end_evaluation = df_idx[train_num+26]
        evaluate_data = data.loc[df_idx[train_num+1]: end_evaluation, "gw-level"].values


        # tune for best hyperparamters and evaluate on MAPE metric
        best_params = search_hyperparameters(data, end_train, end_valid, exog_features, transformer_exog)

        # train and make predict into 26 months in the future of the test template
        df_predictions, smape = train_and_predict(data,best_params, evaluate_data, end_valid, end_train, valid_num, train_num, df_idx, exog_features, transformer_exog)
        df_predictions["pred"] =  df_predictions["pred"].round(2)
        df_predictions.loc["smape", "pred"] = smape


    except Exception as ex:
        print("[Error]")
        print(traceback.format_exc())


    print("SMAPE value: {}%".format(smape))
    print("> FORECAST Done!!!!")


    return df_predictions

def main_forecast():

     args = get_argument()

     location_id = args.location_id
     ##region = args.region
     region_dir = args.region_dir

     df_mps_all = get_all_mps(region_dir)

     df_temp = process_region_gw_temp(region_dir, df_mps_all, location_id, radius=10000)

     df_recursive = recursive_populate_template(df_temp, location_id)

     df_exogenous_features = add_exogenous_features(df_temp, df_recursive, location_id)

     df_preds = populate_test_data(df_exogenous_features)

     df_preds.to_csv(f"predictions-{location_id}.csv")


if __name__ == "__main__":

    main_forecast()
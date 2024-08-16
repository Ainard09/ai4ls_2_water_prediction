import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import select_features
import re
import traceback




def search_space(trial):
  """ Regressor hyperparameters search space"""

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


def search_hyperparameters(data, end_train, end_valid):

  # instantiate a forcaster transformer with categorical features
  forecaster = ForecasterAutoreg(
  regressor = HistGradientBoostingRegressor(
                  random_state=123),
  lags = 24
  )

  results_search, _ = bayesian_search_forecaster(
  forecaster         = forecaster,
  y                  = data.loc[:end_valid, 'temp'],
  search_space       = search_space,
  steps              = 36,
  refit              = False,
  metric             = 'mean_absolute_percentage_error',
  initial_train_size = len(data.loc[:end_train]),
  fixed_train_size   = False,
  n_trials           = 7,
  random_state       = 123,
  return_best        = True,
  n_jobs             = 'auto',
  verbose            = False,
  show_progress      = True
  )

  best_params = results_search['params'].iat[0]

  return best_params

def recursive_train_predict(data, best_params, end_valid):

  # train for future predictions
  forecaster = ForecasterAutoreg(
  regressor = HistGradientBoostingRegressor(**best_params,
                  random_state=123),
  lags = 24
  )
  # train the model the time series train and validation dataset
  forecaster.fit(
    y    = data.loc[:end_valid, 'temp']
  )

  # make predictions into the future
  predictions = forecaster.predict(
    steps    = 26
  )
  df_preds = pd.DataFrame(predictions)

  return df_preds


def recursive_populate_template(df, hrbnz01):

    print("> Start Recursive Forecast for Temperature....")

    try:

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')

        if "2021-11-01" not in df.index:
          print(f"The location ID {hrbnz01} time series is incomplete without ending on 2021-12-01 but it ends on {df.index[-28]}")
          exit()
        
        else:
          data = df.copy()
          df_idx = data.index
          train_num = int(len(data.loc[: df_idx[-28]]) * 0.8)
          valid_num = len(data.loc[: df_idx[-28]])
          end_train = df_idx[train_num]
          end_valid = df_idx[valid_num]
          end_evaluation = df_idx[train_num+36]
          evaluate_data = data.loc[df_idx[train_num+1]: end_evaluation, "temp"].values

          # tune for best hyperparamters and evaluate on MAPE metric
          best_params = search_hyperparameters(data, end_train, end_valid)

          # train and make predict into 26 months in the future of the test template
          df_predictions = recursive_train_predict(data, best_params, end_valid)
          df_predictions["pred"] =  df_predictions["pred"].round(2)
          df_predictions.reset_index(inplace=True)
          df_predictions.columns = ["date", hrbnz01]
      

    except Exception as ex:
        print("[Error]")
        print(traceback.format_exc())

    print("> Recursive Done.....")

    return df_predictions


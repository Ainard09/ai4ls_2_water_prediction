## Data preprocessing
**Notebook**: [preprocess_ehyd_data.ipynb](preprocess_ehyd_data.ipynb)

**Notes**:

For each measurement point in each region, create a summary file that contains (yr_max, yr_min, monthly_gw, temp - if available -,) in addition to some other features (e.g. yr_avg, yr_max_dist, yr_min_dist).

## Plotting gw-level & Searching for Patterns
**Notebook**: [plot_gw_levels_and_explore_patterns.ipynb](plot_gw_levels_and_explore_patterns.ipynb)

**Notes**:

Plotting the gw-level for different measurement points (mps) to look for seasonal data, trends, ..etc.

## Add Exogenous Features
**Notebook**: [add_exogenous_features.ipynb](add_exogenous_features.ipynb)

**Notes**:

Examples: quarter, season, weather, cyclical_features.

## Explore and Plot Temperature Data
**Notebook**: [explore_and_plot_temperature_data.ipynb](explore_and_plot_temperature_data.ipynb)


## Forecast missing temperature values backward
**Notebook**: [forecast_temperature_backward.ipynb](forecast_temperature_backward.ipynb)

**Notes**:

Process the few existing temp data to forecast missing values backwards.

## Compare Different ML models on Train / Val sets to Pick the Best Model
**Notebook**: [compare_models_on_train_val_set.ipynb](compare_models_on_train_val_set.ipynb)

**Notes**:

* Train / Val Split Percentage: Train set 80% / Val set 20%
* Compared regressors: XGBoost, LightGBM, CatBoost, HistGradientBoost
* Chosen regressor: HistGradientBoost with these [SMAPE scores](smape_scores.csv).

## Final Forecast Notebook for the 487 mps in the .csv template
**Notebook**: [forecast_final_487_data_points.ipynb](forecast_final_487_data_points.ipynb)

**Notes**:

Using HistGradientBoostRegressor. Kindly [click here for forecasting results](gw_test_results.csv).

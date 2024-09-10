## Motivations

The sole objective of this projective is to build a machine learning pipeline designed to forecast monthly average groundwater levels of given geographical locations in Austria with 26 months horizon into the future.

## Dataset

The groundwater data of nine regions in Austria are provided from a public metereological [website](https://ehyd.gv.at/#). These nine regions include; `Burgenland`, `Kärnten`, `Niederösterreich`, `Oberösterreich`, `Salzburg`, `Steiermark`, `Tirol`, `Vorarlberg`, `Wien`. There are other information relating to groundwater in each region file, however, we are only focusing on few datasets that align with our goal of building a robust forecaster model for monthly groundwater levels. These are the following dataset we select:

- Grundwasserstand-Monatsmittel
- Grundwassertemperatur-Monatsmittel
- Stammdaten
- messstellen_alle.csv

The given 487 geographical locations test data were selected from aforemention regions and dataset above.

## Brainstorming

We took a holistic approach in defining a system for this project. It's important to note that this is a time series analysis project, and the consistency of the historical data is crucial. As such, we dove deeper to understand the dataset. Secondly, we are working with 487 datasets that limit us from taking a deeper insight into individual data. The seasonal patterns and trends are quite different in most locations, while a universal model must be defined to process all locations at the same time.

We noticed that not all given locations' data has corresponding monthly temperature data. Interestingly, research has shown that groundwater temperatures tend to be relatively consistent over short distances due to the thermal properties of aquifers and the slow movement of groundwater. Therefore, we devised a method by defining a radius of 10,000 meters away from the measuring point of the given location and randomly selecting the nearest temperature data (we did not restrict ourselves to only 487 sites but used the entire dataset for searching).

While using temperature data as one of the exogenous features, we do not have data for the 26 horizons into the future. We addressed this by recursively forecasting all 26 horizons for all 487 locations before going on to forecast the groundwater. The following illustrates the steps we took in forecasting groundwater values for 26 months into the future for all 487 geographical locations in Austria.

## Data preprocessing

**Notebook**: [preprocess_gw_temperature.ipynb](preprocess_gw_temperature.ipynb)

**Notes**:

The 487 locations were extracted from the entire datasets and preprocessed the groundwater and temperature values. As stated earlier, sites without corresponding temperatures were assigned temperatures from neighbouring location.

## Plotting gw-level & Searching for Patterns

**Notebook**: [plot_gw_levels_and_explore_patterns.ipynb](plot_gw_levels_and_explore_patterns.ipynb)

**Notes**:

Plotting the gw-level for different measurement points (mps) to look for seasonal data, trends, ..etc.

## Temperature Recursive Forecast

**Notebook**: [forecast_temperature_backward.ipynb](forecast_temperature_backward.ipynb)

**Notes**:

Recursively forecast the 26 months for all the givien test data.

## Add Exogenous Features

**Notebook**: [add_exogenous_features.ipynb](add_exogenous_features.ipynb)

**Notes**:

Calender and climate features were introduced to effectively help the model to capture underlying patterns in the dataset: **month, year, quarter, season, weather, cyclical_features**. The weather data were sourced from

Similarly, the rolling mean, max, min for temperature were also included to retain correlation and patterns within the temperature values as we progress into the future. Moving forward, we defined feature interactions amongs the exogenous features to help the model capture the patterns between features.

## Compare Different ML models on Train / Val sets to Pick the Best Model

**Notebook**: [compare_models_on_train_val_set.ipynb](compare_models_on_train_val_set.ipynb)

**Notes**:

- Train / Val Split Percentage: Train set 80% / Val set 20%
- Compared regressors: XGBoost, LightGBM, CatBoost, HistGradientBoost
- Chosen regressor: HistGradientBoost has the lowest the smape score, indicating better performance metric compare to other.

## Final Forecast Notebook for the 487 geographical locations in the .csv template

**Notebook**: [forecast_final_487_data_points.ipynb](forecast_final_487_data_points.ipynb)

**Notes**:

Using HistGradientBoostRegressor. The populated template and smape score results are stored in the results folder.

## Summary

Overall, the forecast model shows high performaning metric values on Symmentric Mean Percentage Error (SMAPE) for all the 487 geographical locations. The SMAPE values are all below 1%, indicating high accuracy of the model in predicting the 26 horizons. Despite the differences in irregularity of seasonal trends, correlations of groundwater levels, the machine learning pipeline could take any data (location) within the ehdy dataset with time series values of groundwater levels up until 2021 and give robust forecast into the future.

Future investigations would be to look into climate data that could help the model capture patterns and improve forecast accuracy. Although, the dearth of temperature data on some locations forced us to randomly search for neighbouring area temperature values to help improve the model's performance. Availability of other climate data (e.g precipitation, surface temperature) for all the geographical locations will have profound impact on the model performance.

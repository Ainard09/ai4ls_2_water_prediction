## Installation and Usage

### Dataset and File

Have all the 9 regions (if you choose is to work with a single or selected regions, please proceeds with the same step) datasets in a **folder** with the corresponding sub-folders. These are the required files for each region:

- **Grundwasserstand-Monatsmittel** - folder
- **Grundwassertemperatur-Monatsmittel** - folder
- **messstellen_alle.csv** - file

Secondly, have all the py files in the same directory root: `forecast.py`, `add_exogenous.py`, `preprocess_temp.py`, `recursive_temp_pred.py`, `use_functions.py`

## Dependencies

Create a python virtual environment and `pip install skforecast scikit-learn` within the venv.

## Run Code

The code simply takes the location ID and the directory to the location ID in the data directory, saving the output predictions of 26 horizons into the future in a csv file.

From cmd line, run the `forecast.py` follow by other two arguments.
A typical example:

- `python forecast.py "306985" "/../../../data/ehyd_messstellen_4"`

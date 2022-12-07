""" Convert Inputted NetCDF file into CSV file format """

import xarray as xr
import pandas as pd

fileName = "name_of_file"
data = xr.open_dataset(fileName + ".nc")
data_df = data.to_dataframe().reset_index()

data_df.to_csv("results_" + fileName + ".csv")
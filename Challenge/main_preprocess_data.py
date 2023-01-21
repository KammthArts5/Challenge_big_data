from def_preprocess_data import *   
import numpy as np
import pandas as pd

#%% Fill missing data
data = import_csv_data("factory_process_KC12 KC12.csv")
heads = data.columns.values
data2 = fill_missing_data(data.to_numpy())

export_excel_data(pd.DataFrame(data2, columns=heads), "factory_process_KC12 KC12_filled.xlsx")
from def_preprocess_data import *   
import numpy as np
import pandas as pd

#%% test
data = import_excel_data("test.xlsx")
heads = data.columns.values
data2 = fill_missing_data(data.to_numpy())
#%%

export_excel_data(pd.DataFrame(data2, columns=heads), "clean_test.xlsx")
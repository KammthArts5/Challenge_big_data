import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

#%% Files function
def import_excel_data(excel_file_name):
    '''
    Import data from an Excel file.

    Parameters
    ----------
    excel_file_name : String
        Name of the Excel file to extract.

    Returns
    -------
    DataFrame object
        Data from the Excel file in a Pandas DataFrame object.

    '''
    return pd.read_excel(excel_file_name)

def import_csv_data(csv_file_name, sep=';'):
    '''
    Import data from a CSV file.

    Parameters
    ----------
    csv_file_name : String
        Name of the CSV file to extract.

    Returns
    -------
    DataFrame object
        Data from the Excel file in a Pandas DataFrame object.

    '''
    return pd.read_csv(csv_file_name,sep=sep)

def export_excel_data(data_to_export, excel_file_name, sheet_name="Sheet1"):
    '''
    Export data in an Excel file.

    Parameters
    ----------
    data_to_export : Pandas DataFrame
        Data to export in an Excel file.
    excel_file_name : String
        Name of the Excel file to export the data.
    sheet_name : String, optional
        Name of the sheet in the Excel file to export the data. The default is "Sheet1".

    Returns
    -------
    None.

    '''
    
    writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
    data_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    
#%% Preprocess functions 
def fill_missing_data(data):
    '''
    Fill missing data using KNN Imputer.

    Parameters
    ----------
    data : Numpy Array
        Array to fill.

    Returns
    -------
    Numpy Array
        Filled array.

    '''
    
    imputer = KNNImputer(n_neighbors=2,weights="uniform")
    
    return imputer.fit_transform(data)


    
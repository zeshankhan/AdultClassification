import xlrd
import pandas as pd
import os
def get_dataset(loc="../input/adultData.xlsx"):
    xls = pd.ExcelFile(loc)
    training = pd.read_excel(xls, 'training',index_col=None)
    testing = pd.read_excel(xls, 'testing',index_col=None)
    #df1 = pd.read_excel(xls, 'table')
    return training, testing#, df1

#d1,d2,d3=get_dataset(loc="../input/adultData.xlsx")
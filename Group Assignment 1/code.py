import pandas as pd
import numpy as np
import openpyxl as px

# load in the basic data
basic_data = pd.read_excel('basic_data.xlsx')

# how many observations are there
len(basic_data) # 245

# what are the attributes here?
str(basic_data)


# generating summary statistics of t
basic_data.describe()



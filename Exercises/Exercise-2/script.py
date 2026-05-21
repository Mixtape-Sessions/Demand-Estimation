"""Script template."""

import pyblp
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf



pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

import IPython.display
IPython.display.display(IPython.display.HTML('<style>pre { white-space: pre !important; }</style>'))


# You can read the product data directly from its URL.
product_data = pd.read_csv('https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/products.csv')

##
data=pd.read_csv('C:\\Users\\micha\\OneDrive\\Documents\\Mixtape Sessions\\Demand-Estimation\\Exercises\\Data\\products.csv')
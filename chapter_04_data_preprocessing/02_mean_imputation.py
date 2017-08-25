import pandas as pd
from sklearn.preprocessing import Imputer

df = pd.read_csv('../data/dropna_example.csv')

""" Mean Imputation:
    Replace NaN with mean from column
"""
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)

imputed_data = imr.transform(df.values)
print(imputed_data)
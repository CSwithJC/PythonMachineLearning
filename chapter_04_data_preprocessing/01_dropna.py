import pandas as pd

df = pd.read_csv('../data/dropna_example.csv')
print('Dataframe from CSV:')
print(df)

# get bool array that says if the value is NaN or not
print('\nBool array that states if each value is a number or not:')
print(df.isnull())

# get the number of NaN values per column
print('\nNumber of NaN values per column:')
print(df.isnull().sum())

# convert dataframe to nd-array:
print('\nConvert df to ndarray:')
print(df.values)

# drop all rows with NaNs from the dataset:
print('\nDrop all rows with NaNs from dataframe')
print(df.dropna())

# drop all columns with NaNs from dataset:
print('\nDrop all columns with NaNs from dataframe')
print(df.dropna(axis=1))

# drop rows where all columns are NaN:
print('\nDrop rows where all columns are NaN:')
print(df.dropna(how='all'))

# drop rows that have not at least 4 non-NaN vals:
print('\nDrop rows that have not at least 4 non-NaN vals:')
print(df.dropna(thresh=4))

# drop rows where NaNs appear in a specific column:
print('\nDrop rows where NaNs appear in a specific column:')
#print(df.dropna(subset=['C']))

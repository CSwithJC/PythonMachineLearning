import pandas as pd

df = pd.read_csv('../data/iris.csv', header=None)
print(df.tail())
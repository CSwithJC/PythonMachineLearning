import pandas as pd

from sklearn.model_selection import train_test_split

"""It is important to partition the dataset
   into training and testing data. A convenient
   way of doing this is using cross_validation:
"""
df_wine = pd.read_csv('../data/wine.csv')
df_wine.columns = ('Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash',
                   'Magnesium',
                   'Total phenola',
                   'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline')


print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)



import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

"""Bringing features onto the same scale is of EXTREME importance.
   Most learning algorithms (except Decision Trees and Random Forests)
   behave much better if the data is on the same scale.

   Two common methods for putting features to the same scale are:

   - Normalization = puts values in the range [0, 1]
   - Standarization = makes mean 0 and standard dev 1, making the data
                      behave like a Normal Distribution. This is preferred
                      because it makes learning the weights easier and still
                      holds data from the outliers, making it less sensitive
                      to outliers in the future.
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


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)

"""Normalization: """
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

"""Standarization:"""
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
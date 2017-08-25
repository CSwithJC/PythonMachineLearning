import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

"""One-Hot Encoding:

   For nominal features (t-shirt color), converting
   the strings to just numbers is a bad idea because
   the learning algorithm will believe that one is
   larger than the other (this is entirely wrong).

   A common way to work around this is using
   One-Hot Encoding, which returns a sparse matrix
   and converts the strings to "binary" values. For
   example:

   blue | green | red
   _____|_______|_____
    0   |   1   |  0
    0   |   0   |  1
    1   |   0   |  0

    One-hot encoding can be done using both skikit-learn
    and Pandas. The first part of the code is the same
    as before, but skip down for the One-hot Encoding part:

"""

# Dataframe with both types of features:
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

# Set the column names of the dataframe
df.columns = ['color', 'size', 'price', 'classlabel']

# Create mapping for shirt sized
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}

# Map the values inside the df using size_mapping
df['size'] = df['size'].map(size_mapping)

# To do a reverse mapping of size, use the following:
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# Enumerate labels (Order doesn't matter!)
class_mappings = {
    label: idx for idx, label in enumerate(np.unique(df['classlabel']))
}
# Now use the mapping dictionary:
df['classlabel'] = df['classlabel'].map(class_mappings)


"""------ One-hot Encoding -------"""
# For One-hot encoding to work, you must first encode the
# values into numbers like we did before:
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

# Put categorical_features = 0 to let it know that you
# want to encode just the first column (color).
ohe = OneHotEncoder(categorical_features=[0])

print('\nAfter doing One-hot Encoding, the dataframe is:')
print(ohe.fit_transform(X).toarray())


# This can also be done in Pandas using get_dummies
# pd.get_dummies(df[['price', 'color', 'size']])

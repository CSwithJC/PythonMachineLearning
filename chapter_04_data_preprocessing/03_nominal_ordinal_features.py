import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

"""Types of features:

Ordinal: Can have an order (T-shirt size; XL>L>M
Nominal: Have no order (T-shirt color: blue !> red)

"""

# Dataframe with both types of features:
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

# Set the column names of the dataframe
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

# Create mapping for shirt sized
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}

# Map the values inside the df using size_mapping
df['size'] = df['size'].map(size_mapping)

print('\nDataframe after mapping sizes:')
print(df)

# To do a reverse mapping of size, use the following:
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# Enumerate labels (Order doesn't matter!)
class_mappings = {
    label: idx for idx, label in enumerate(np.unique(df['classlabel']))
}

print('\nMapping for the labels:')
print(class_mappings)

# Now use the mapping dictionary:
df['classlabel'] = df['classlabel'].map(class_mappings)
print('\nDataframe after mapping class labels:')
print(df)

# To do a reverse mapping of label, use the following:
inv_size_label = {v: k for k, v in class_mappings.items()}
df['classlabel'] = df['classlabel'].map(inv_size_label)
print('\nDataframe after inversing mappedclass labels:')
print(df)

# Alternately, sklearn's LabelEncoder does the same:
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print('\nThe same can be done using sklearn\'s LabelEncoder')
print(y)
print(class_le.inverse_transform(y))
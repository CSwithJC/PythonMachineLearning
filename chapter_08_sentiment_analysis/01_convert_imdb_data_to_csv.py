import os
import numpy as np
import pandas as pd

create_csv_file = False

if create_csv_file:
    labels = {'pos': 1, 'neg':0}

    df = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = '../data/aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)

    df.columns = ['review', 'sentiment']

    np.random.seed(0)

    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('../data/movie_data.csv', index=False)
    df.head(3)

df = pd.read_csv('../data/movie_data.csv')
print(df.head(3))
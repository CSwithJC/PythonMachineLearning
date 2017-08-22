import pandas as pd
import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()).join(emoticons).replace('-', '')
    return text

df = pd.read_csv('../data/movie_data.csv')
df['review'] = df['review'].apply(preprocessor)
print(df.head())
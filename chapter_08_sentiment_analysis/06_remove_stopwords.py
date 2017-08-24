from nltk.stem.porter import PorterStemmer
import pandas as pd
import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()).join(emoticons).replace('-', '')
    return text

df = pd.read_csv('../data/movie_data.csv')
df['review'] = df['review'].apply(preprocessor)
print(df.tail())


def tokenizer(text):
    return text.split()

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lit')[-10:] if w not in stop])
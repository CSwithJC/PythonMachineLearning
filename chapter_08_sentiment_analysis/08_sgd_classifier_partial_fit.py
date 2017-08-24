import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

stop = stopwords.words('english')


# Clean data, remove stopwords, and return tokenized words
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()).join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


# Generator function that reads in and returns one doc at a time
def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# Takes a document and returns a particular number of documents
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# HashingVectorizer uses the 32-bit MurmurHash3 Algorithm
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log',
                    random_state=1,
                    n_iter=1)

doc_stream = stream_docs(path='../data/movie_data.csv')

# Start the out-of-core learning
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

# Not as efficient as with grid search, but this is much more
# memory-efficient
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# update the model:
clf = clf.partial_fit(X_test, y_test)
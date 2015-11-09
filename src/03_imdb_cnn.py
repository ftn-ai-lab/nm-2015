from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

import cPickle as pickle
from nltk import word_tokenize

'''
    Sentiment analiza IMDB recenzija sa 1D konvolutivnom neuronskom mrezom.
    Sentiment analiza - klasifikacija sentimenta na pozitivan i negativan.

    Tacnost klasifikacije ~82% posle 3 epohe. 18 sekundi po epohi na NVidia GTX 770.
'''

# utility funkcija za deserijalizaciju
def unpickle(path):
    f = open(path, 'rb')
    d = pickle.load(f)
    f.close()
    return d

# parametri
max_features = 10000 # koliko reci iz celog recnika uzimati u obzir
maxlen = 100 # max duzina recenzije
batch_size = 32
embedding_dims = 100
nb_filter = 250 # broj konvolutivnih 1D filtera
filter_length = 3 # duzina konvolutivnog 1D filtera
hidden_dims = 250 # broj neurona u skrivenom sloju
nb_epoch = 3

print("Loading data...")
# ucitavanje IMDB recenzija
# Recenzije su preprocesirane da budu sekvenca indeksa reci.
# Reci su indeksirane po broju ponavljanja, tako da sto je manji indeks - to se ta rec cesce ponavlja u celom datasetu.
# Ovi indeksi se krecu od 2, tj. indeks 2 oznacava je najcescu rec, 3 sledecu, itd.
# Indeks 0 je rezervisan za tzv. OOV (out-of-vocabulary) reci, tj. reci koje se ne nalaze u recniku
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, start_char=None,
                                                        oov_char=0,
                                                        index_from=0,
                                                        test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# "sequence padding"
# Ako su recenzije duze od maxlen, odseci ih da budu duzine maxlen.
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# Prvi sloj je "embedding" sloj, tj. mapiranje ulaznog vektora (sekvenca indeksa reci) na "dense" vektor fiksnih dimenzija
# max_features je broj "feature"-a, odnosno reci, embedding_dims je dimenzija rezultujuceg "dense" vektora
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.25))

# 1D konvolutivni filter
# grupisanje reci po grupama od filter_length
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        activation="relu"))
                        
# standardni MAX-pooling sloj koji smanjuje broj parametara na pola (pool_length=2)
model.add(MaxPooling1D(pool_length=2))

# Flatten() prethodnih konvolutivnih slojeva kako bismo mogli spojiti na sledeci Dense sloj
model.add(Flatten())

# standardni skriveni sloj
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# na izlazu samo jedan neuron sa sigmoidalnom aktivacionom funkcijom
# izlaz 1 = pozitivna recenzija, izlaz 0 = negativna recenzija
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_data=(X_test, y_test))

# recnik u formatu {key=rec, value=indeks_reci}
dictionary = unpickle('pkl/imdb.dict.pkl')
# pravljenje inverznog recnika {key=indeks_reci, value=rec}
dictionary_reverse = dict(zip(dictionary.values(), dictionary.keys()))

# utility funkcija koja "dekodira" recenziju 
# od vektora sa indeksima reci u prave reci
def decode(x):
    review = ''
    for w in x:
        if w == 0:
            review += ' OOV'
            continue
        review += ' ' + dictionary_reverse[w]
    return review[1:]

# utility funkcija koja "enkodira" recenziju
# od reci u vektor sa indeksima reci 
def encode(review):
    review = review.lower()
    tokens = word_tokenize(review)
    x = []
    for token in tokens:
        token_enc = dictionary[token]if dictionary.has_key(token) else 0
        if token_enc > max_features:
            token_enc = 0
        x.append(token_enc)
    x = np.array(x)
    return x

# utility funkcija koja za datu recenziju (tekst) predvidja da li je negativna ili pozitivna
def predict_review(review):
    x = encode(review)
    X = np.array([x])
    X = sequence.pad_sequences(X, maxlen=maxlen)
    y = model.predict(X)
    return y[0][0]
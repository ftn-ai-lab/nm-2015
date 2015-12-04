from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import sys
import os

'''
    Generisanje emailova studenata (-ish).
    Potrebno bar XX epoha pre nego što generisani emailovi počnu da dobijaju smisla.

    Potrebno bar 100k karaktera za obučavanje, 1M bi bilo poželjnije,
    (dataset sa studenskim emailovima ima oko 180k karaktera).
    
    Obučavanje po epohi ~140s na NVidia GTX 770.
'''

# ucitavanje emailova
# emailovi su u formatu '%start% tekst emaila %end%', svaki red jedan email
# %start% i %end% su placeholderi za pocetak i kraj emaila
path = 'emails.txt'
text = open(path).read()

# uklanjanje svih non-utf8 karaktera
text = text.decode('utf-8', 'ignore').encode("utf-8")

print('corpus length:', len(text))

text_split = text.split('\n')

# pravljenje skupa svih karaktera
chars = set(text)
print('total chars:', len(chars))
# dictionary karakter -> indeks 
char_indices = dict((c, i) for i, c in enumerate(chars))
# dictionary index -> karakter
indices_char = dict((i, c) for i, c in enumerate(chars))

# uzimanje recenica po 20 karaktera iz teksta sa korakom 1
maxlen = 20
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    # uzimanje maxlen karaktera
    sentences.append(text[i: i + maxlen])
    # predvidjanje sledeceg karaktera na osnovu maxlen prethodnih
    next_chars.append(text[i + maxlen])
    
print('nb sequences:', len(sentences))

# pravljenje obucavajuceg skupa
# vektorizacija: X: recenice sa maxlen karaktera -> vektor dimenzija (maxlen, len(chars))
# vektorizacija: y: slovo -> vektor dimenzija (len(chars))
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# model: 2 LSTM sloja
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# utility funkcija za uzorkovanje iz niza verovatnoca
# temperatura > 1: raspodela verovatnoca sve vise tezi ka uniformnoj
# temperatura < 1: raspodela verovatnoca sve vise tezi ka cistoj multinomijalnoj
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# na svakih 5 epoha (1 iteracija) snimaju se parametri/tezine
nb_epoch = 60
epoch_per_iter = 5

# da li sacuvati tezine pri obucavanju
save_weights = True
# da li ucitati tezine (za testiranje, bez obucavanja)
load_weights = True

def strip(email):
    email = email.replace('%start%', '')
    email = email.replace('%end%', '')
    return email

def generate(prime, temperature, length = 400, verbose=False):
    sentence_rnd = prime if prime is not None else np.random.choice(text_split)[:maxlen]
    generated = ''
    sentence = sentence_rnd[:] # kopiranje stringa
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')

    # generisanje 400 karaktera
    for it in range(length):
        # vektorizacija
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        # predikcija sledeceg karaktera
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        
        # konkatenacija rezultata za sledecu iteraciju u generisanju
        generated += next_char
        sentence = sentence[1:] + next_char
        
        # %end% je placeholder za kraj emaila
        # ako mreza izgenerise %end%, prekinuti generisanje
        if '%end%' in generated:
            break

        if verbose:
            sys.stdout.write(next_char)
            sys.stdout.flush()
    
    return generated

weights_path_prefix = 'weights/lstm_email_weights_'
weights_path_suffix = '.hdf5'

# obucavanje LSTM
def train_lstm():
    for iteration in range(1, (nb_epoch / epoch_per_iter) + 1):
        
        weights_path = weights_path_prefix + str(iteration) + weights_path_suffix
        
        # preskoci iteraciju za koju nema tezina
        if load_weights and not os.path.exists(weights_path):
            continue
        
        print()
        print('-' * 50)
        print('Iteration', iteration)
        
        if load_weights:
            # ucitavanje tezina za iteraciju
            model.load_weights(weights_path)
        else:
            # obucavanje - 5 epoha
            model.fit(X, y, batch_size=128, nb_epoch=epoch_per_iter)
        
        # snimanje tezina za iteraciju
        # ako 
        if save_weights and not load_weights:
            model.save_weights(weights_path)
        
        # pocetak nasumicne recenice za tzv. "priming"
        sentence_rnd = np.random.choice(text_split)[:maxlen]
        
        # prikaz generisanih emailova za razne temperature uzorkovanja
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- temperature:', temperature)
            
            print(strip(generate(sentence_rnd, temperature)))
            
            print()

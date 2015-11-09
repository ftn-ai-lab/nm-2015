from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, AutoEncoder
from keras.callbacks import Callback
from keras.utils import np_utils

'''
    MLP autoencoder za MNIST primer.
'''

# parametri
batch_size = 128
nb_classes = 10
nb_epoch = 80

# podaci: skup za obucavanje (60k uzoraka) i skup za validaciju/testiranje (10k uzoraka)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape iz matrice 28x28 u vektor sa 784 elemenata
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# pretvaranje u float zbog narednog koraka
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# skaliranje na [0,1]
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# konvertuj klase u one-hot encoding vektore
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# definisanje autoenkodera
# prvo enkoder: 784-128-32
encoder = Sequential()
encoder.add(Dense(128, input_shape=(784,)))
encoder.add(Activation('tanh'))
encoder.add(Dense(32))
encoder.add(Activation('tanh'))

# zatim dekoder: 32-128-784
decoder = Sequential()
decoder.add(Dense(128, input_shape=(32,)))
decoder.add(Activation('relu'))
decoder.add(Dense(784))
decoder.add(Activation('relu'))

# povezivanje enkodera i dekodera u autoenkoder
model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

model.compile(loss='mse', optimizer='rmsprop')

# utility klasa iscrtavanje rezultata preko callback-a
class TestCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        test_on_samples(3, False)
        
# utility funkcija za iscrtavanje rezultata: ulazna slika -> rekonstruisana slika
def test_on_samples(nb_samples=10, random=True):
    for i in range(nb_samples):
        idx = np.random.randint(X_test.shape[0]) if random == True else i
        input_vect = X_test[idx]
        input_img = input_vect.reshape(28,28)
        output_vect = model.predict(np.array([input_vect]))
        output_img = output_vect.reshape(28,28)
        fig = plt.figure(figsize=(3, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(input_img, 'gray')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(output_img, 'gray')
        plt.show()

# obucavanje neuronske mreze
model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(X_test, X_test), callbacks=[TestCallback()])
        
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

'''
    Jednostavna vestacka neuronska mreza za MNIST dataset
'''

# broj primeraka za SGD
batch_size = 128

# broj izlaza (klasa) - 10 cifara
nb_classes = 10

# broj epoha
nb_epoch = 20

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

# definisanje vestacke neuronske mreze
# 784 neurona na ulazu, 128 u skrivenom sloju, 10 na izlazu
# sigmoidalna logisticka aktivaciona funkcija
model = Sequential()
model.add(Dense(128, input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))

# definisanje SGD, lr je learning rate
sgd = SGD(lr=0.01)

# kompajliranje modela (Theano) - optimizacija svih matematickih izraza
model.compile(loss='mse', optimizer=sgd)

# obucavanje neuronske mreze
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_data=(X_test, Y_test))

# nakon obucavanje testiranje
score = model.evaluate(X_test, Y_test, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range

import numpy as np
import matplotlib.pyplot as plt

'''
    Jednostavna duboka konvolutivna neuronska mreza za CIFAR10 dataset.
    CIFAR10 dataset se sastoji od 60k prirodnih slika podeljenih u 10 kategorija:
    - avion
    - automobil
    - ptica
    - macka
    - jelen
    - pas
    - zaba
    - konj
    - brod
    - kamion
    
    CNN je obucena u 99 epoha sa tacnoscu klasifikacije ~80%.
    Obucavanje je trajalo oko 6h, odnosno oko 220 sekundi po epohi na NVidia GTX 770.
'''

batch_size = 32
nb_classes = 10 # 10 klasa/kategorija
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
nb_epoch = 100
data_augmentation = True # da li normalizovati i poboljsati podatke

save_weights = True # snimati tezine u svakoj iteraciji obucavanja
test_only = True # da li samo koristiti vec obucenu mrezu

# CIFAR10 slike su 32x32
img_rows, img_cols = 32, 32
# CIFAR10 slike imaju RGB kanale
img_channels = 3

# podaci podeljeni na podatke za obucavanje (50k) i testiranje (10k)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# konvertuj klase u one-hot encoding vektore
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

# prvi conv sloj se sastoji od 32 filtera 3x3, ulaz su slike 32x32x3
# aktivaciona funkcija je ReLU
model.add(Convolution2D(32, 3, 3, border_mode='full',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # MAX-pooling sa filterom 2x2
model.add(Dropout(0.25)) # sansa za dropout 25%

# drugi conv sloj je 64 filera 3x3
# aktivaciona funkcija je ReLU
model.add(Convolution2D(64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # MAX-pooling sa filterom 2x2
model.add(Dropout(0.25)) # sansa za dropout 25%

# na kraju kao klasifikator - FC MLP (fully connected MLP)
model.add(Flatten()) # izlaz iz prethodnih slojeva je matrica, Flatten() ga pretvara u vektor
model.add(Dense(512)) # 512 neurona u skrivenom sloju
model.add(Activation('relu'))
model.add(Dropout(0.5)) # sansa za dropout 50%
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# SGD + momentum + Nesterov momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# ucitavanje snimljenih tezina iz prethodnog obucavanja
if test_only:
    model.load_weights('weights/cifar10_cnn.hdf5')

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
    
if not data_augmentation:
    print("Not using data augmentation or normalization")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    best_score = 1000
    best_epoch = 0

    for e in range(nb_epoch):
        if test_only:
            continue
        
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score, acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0], values=[("test loss", score), ("accuracy", acc)])
        
        if save_weights and score < best_score:
            best_score = score
            best_epoch = e
            model.save_weights('weights/cifar10_cnn.hdf5', overwrite=True)
        
        print("Best epoch", best_epoch)

def test_rand_img():
    '''Pomocna funkcija za testiranje CIFAR10 CNN
    1. Nasumicna slika iz test skupa
    2. Konverovanje slike 3x32x32 -> 32x32x3
    3. Dovodjenje slike na ulaz CNN (uz preprocesiranje)
    '''
    # random slika iz test skupa
    rand_index = np.random.randint(0, X_test.shape[0])
    X_test_img = X_test[rand_index]
    # prvo mora konvert iz 3x32x32 -> 32x32x3
    img = np.ndarray((img_rows, img_cols, img_channels))
    for i in range(img_rows):
        for j in range(img_cols):
            img[i, j, 0] = X_test_img[0, i, j]
            img[i, j, 1] = X_test_img[1, i, j]
            img[i, j, 2] = X_test_img[2, i, j]

    plt.figure()    
    plt.imshow(img)

    # preprocesiranje
    if data_augmentation:
        output = model.predict(np.array([datagen.standardize(X_test_img)]))
    else:
        output = model.predict(np.array([X_test_img]))
        
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(nb_classes), output[0], align='center')
    plt.xticks(np.arange(nb_classes), classes)
    
    print(output)
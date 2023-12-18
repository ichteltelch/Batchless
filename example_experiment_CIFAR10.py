#!/usr/bin/env python
# coding: utf-8

# https://github.com/ichteltelch/Batchless


#get_ipython().system('nvidia-smi')
import tensorflow as tf


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
import math
import numpy as np
import keras

from batchless_normalization import BatchlessNormalization
from batchless_normalization import init_bln_singlePass

from tensorflow.keras.utils import to_categorical


def makeBatchNormalizationLayer(input_shape=None):
    #return BatchNormalization(synchronized=True, input_shape=input_shape)
    if input_shape==None:
        return SyncBatchNormalization()
    else:
        return SyncBatchNormalization(input_shape=input_shape)


"""
Construct the network using the given normalization scheme (may be 'None', 'bn', 'abs', 'log', 'inv')
"""
def makeNetwork(normalization='log'):
    parameterization = normalization
    layers = [];
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer(input_shape=[32,32,3]))
    elif normalization != None:
        BatchlessNormalization(std_parametrization=parameterization, init_std = 1, init_mean=0, input_shape=[32,32,3])
    layers.extend([
        keras.layers.Conv2D(64, 7, activation=None, padding="same", input_shape=[32,32,3]),
        keras.layers.LeakyReLU(),
    ])
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer())
    elif normalization != None:
        layers.append(BatchlessNormalization(std_parametrization=parameterization))
    layers.extend([
        keras.layers.Dropout(0.25),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 5, activation=None, padding="same"),
        keras.layers.LeakyReLU(),
    ])
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer())
    elif normalization != None:
        layers.append(BatchlessNormalization(std_parametrization=parameterization))
    layers.extend([
        keras.layers.Dropout(0.25),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation=None, padding="same"),
        keras.layers.LeakyReLU(),
    ])
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer())
    elif normalization != None:
        layers.append(BatchlessNormalization(std_parametrization=parameterization))
    layers.extend([
        keras.layers.Dropout(0.25),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(50, activation=None),
        keras.layers.LeakyReLU(),
    ])
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer())
    elif normalization != None:
        layers.append(BatchlessNormalization(std_parametrization=parameterization))
    layers.extend([
        keras.layers.Dropout(0.25),
        keras.layers.Dense(50, activation=None),
        keras.layers.LeakyReLU(),
    ])
    if normalization == "bn":
        layers.append(makeBatchNormalizationLayer())
    elif normalization != None:
        layers.append(BatchlessNormalization(std_parametrization=parameterization))
    layers.extend([
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation="softmax")
    ],)
    return keras.models.Sequential(layers)


def main():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data to the range [-1, 1]
    x_train = (x_train.astype('float32') / 127.5 - 1)
    x_test = (x_test.astype('float32') / 127.5 - 1)

    # denormalize the data for testing
    #x_train = x_train *100 + 1000
    #x_test = x_test * 100 + 1000


    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)




    normalizations = [
        None, 
        'bn', 
        'abs', 
        'log', 
        'inv'
    ]
    epochs = 60
    batch_sizes = [
        1,
        2,
        4,8,16,32,
        64,128,256,512,
        1024
        ]
    losses = []
    accuracies = []
    histories = []
    for batch_size in batch_sizes:
        losses.append([])
        accuracies.append([])
        histories.append([])
        for norm in normalizations:
            if norm == 'bn' and batch_size==1:
                # don't attempt to use batch normalizationfor batch size 1
                losses[-1].append(None)
                accuracies[-1].append(None)
                histories[-1].append(None)
            else:
                #create network
                clasificador_cnn = makeNetwork(norm)
                clasificador_cnn.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

                #initialize batch statistics
                init_bln_singlePass(clasificador_cnn, x_train[0:1000])

                #train network
                h = clasificador_cnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.2)

                #save histories
                losses[-1].append(min(h.history['val_loss']))
                accuracies[-1].append(max(h.history['val_accuracy']))
                histories[-1].append(h.history)
                # Print the histories after each test, so we don'l lose results when the program is interrupted
                # and can simply copy a suffix of the output to get all results.
                # Note: That makes the output huge!
                for i, hist_for_bs in enumerate(histories):
                    if hist_for_bs == None:
                        print(f"histories[{i}] is None")
                    else:
                        for j, hist in enumerate(hist_for_bs):
                            if hist == None:
                                pass
                            else:
                                for which in ['val_loss', 'val_accuracy', 'loss', 'accuracy']:
                                    print(f"h['{normalizations[j]}']['{batch_sizes[i]}']['{which}']={hist[which]}")
        #print summaries
        print(f"Norm: {normalizations}")
        print("Loss:")
        for i, line in enumerate(losses):
            print(f"{batch_sizes[i]}: {line}")
            print("Accuracy:")
        for i, line in enumerate(accuracies):
            print(f"{batch_sizes[i]}: {line}")


if __name__ == "__main__":
    main()

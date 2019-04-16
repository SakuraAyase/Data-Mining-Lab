import numpy as np
import load_data
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import matplotlib as mpl
import matplotlib.pyplot as plt



def define_model(input_shape,output_shape):
    model = Sequential()
    model.add(Dense(32,input_dim = input_shape,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(32,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(output_shape,activation="sigmoid"))

    model.compile(loss='categorical_crossentropy',optimizer = RMSprop(lr = 0.001),metrics=['accuracy'])

    return model

def data_process():
    data = load_data.load()[0]
    label = to_categorical(load_data.load()[1] - 1)

    train_size = int(0.8*data.shape[0])
    train_data = data[:train_size,:]
    test_data = data[train_size:, :]
    train_label = label[:train_size,:]
    test_label = label[train_size:,:]

    return train_data,train_label,test_data,test_label

def get_acc(fake,true):
    acc = 0
    f = [np.argmax(one_hot) for one_hot in fake]
    t = [np.argmax(one_hot) for one_hot in true]


    for i in range(len(f)):
        if (f[i] == t[i]):
            acc = acc + 1
    acc = acc / len(fake)

    return acc

def plot_train_log(train_acc,test_acc):
    plt.plot(train_acc,'r')
    plt.plot(test_acc,'b')
    plt.savefig('save.jpg')
    plt.show()


def train(epoches):
    train_data,train_label,test_data,test_label = data_process()
    model = define_model(train_data.shape[1],train_label.shape[1])
    train_acc = []
    test_acc = []
    for i in range(epoches):
        model.fit(train_data, train_label, epochs=1,batch_size=64)

        fake_label_test = model.predict(test_data)
        fake_label_train = model.predict(train_data)


        train_acc.append(get_acc(fake_label_train, train_label))
        test_acc.append(get_acc(fake_label_test, test_label))
    plot_train_log(train_acc,test_acc)


    return model

model = train(10)


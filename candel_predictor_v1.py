import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from DataPrep import *

data_training="market_data/AUD_USD.csv"
data_testing="market_data/AUD_CHF.csv"

def random_data_picker(data,minus,plus):
    picker = np.random.randint(0,len(data))
    hist_data=data[picker-minus:picker]
    imp_value=data[picker]
    epi_data=data[picker+1:picker+plus+1]

    return [hist_data,imp_value,epi_data]


def normalize(inp):
    A = [i*i for i in inp]
    v = (sum(A))**(0.5)
    A = [i / v for i in inp]
    return A

def pick_and_relate(data,minus,plus):
    picker = np.random.randint(0,len(data))
    imp_value = data[picker]
    hist_data=normalize(data[picker-minus:picker]/imp_value)
    epi_data=normalize(data[picker+1:picker+plus+1]/imp_value)
    return [imp_value, hist_data,  epi_data]

class NeuralNetwork():
    def __init__(self):
        self.input = 0
        self.layer1 = 0
        self.layer2 = 0
        self.output = 0
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.save_file = 'default'
        self.metrics = [tf.metrics.BinaryAccuracy(name='accuracy')]
        self.x_in = []
        self.y_in = []
        self.batch_size = 20
        self.epochs = 20
        self.model = tf.keras.Sequential()

    def add_layer(self,Type,Shape,activation,Name,):
        if Type == 'Input':
            self.model.add(tf.keras.layers.Input(shape=(Shape,), name=Name))
        if Type == 'Hidden':
            self.model.add(tf.keras.layers.Dense(shape=(Shape,), activation=activation, name=Name))
        test.input = tf.keras.Input(shape=(45,), name="Input")
        test.layer1 = tf.keras.layers.Dense(32, activation="relu", name="Hidden_1")
        test.layer2 = tf.keras.layers.Dense(32, activation="relu", name="Hidden_2")
        test.output = tf.keras.layers.Dense(4, name="Output")
        self.model.add(tf.keras.layers.Dense(shape=(Shape,),activation=activation,name=Name))



    def Compile(self,SHOW):
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

        if SHOW == True:
            tf.keras.utils.plot_model(self.model, to_file='tmp/'+self.save_file+'.png', show_shapes=True)

    def prep_data(self):
        self.x_train,self.x_val = np.split(np.array(self.x_in), 2)
        self.y_train, self.y_val = np.split(np.array(self.y_in), 2)
        if len(self.x_in) != len(self.y_in) or self.x_train.size != self.x_val.size or self.y_train.size != self.y_val.size or self.x_train.size != self.y_train.size:
            raise AttributeError('Oops training data does not really work')


    def Fit(self,SHOW):
        fit=self.history= self.model.fit(
        self.x_train, self.y_train,
        batch_size = 24,
        epochs = 20,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data = (self.x_val, self.y_val))
        self.history = fit.history

        if SHOW == True:
            plt.title('Loss')
            plt.plot(self.history['loss'], color="coral")
            plt.show()

    def Save(self):
        self.model.save('Neural_Networks/'+self.save_file)

    def predict(self,data,start,end):
        return np.array(self.model.predict(data[start:end]))




test = NeuralNetwork()
test.input=tf.keras.Input(shape=(45,), name="Input")
test.layer1=tf.keras.layers.Dense(32, activation="relu", name="Hidden_1")
test.layer2=tf.keras.layers.Dense(32, activation="relu", name="Hidden_2")
test.output=tf.keras.layers.Dense(4, name="Output")


test.Build(False)
test.Compile()

#print(pick_and_relate(np.array(pd.read_csv('EURmajors/EURUSD_H.csv',usecols=[3])),20,4))



"""
inputs = tf.keras.Input(shape=(45,), name="digits")
x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(2, name="predictions")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.BinaryAccuracy(name='accuracy')])
#tf.keras.utils.plot_model(model, to_file='tmp/NN_1_visual.png', show_shapes=True)# Draw model
print(model.summary())

history=model.fit(
    x_train,y_train,
    batch_size=24,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val,y_val))
hist_dick=history.history

#model.save('Neural_Networks/relation_test')

accuracy = lambda x,y: model.evaluate(x_test[x:y],np.array(y_test[x:y]))
prediction=lambda x,y: np.array(model.predict(x_test[x:y]))



print(str(accuracy(0,100))+'%')
print(prediction(0,10))


plt.title('Loss')
plt.plot(hist_dick['loss'],color="coral")
plt.show()
"""
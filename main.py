import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot



# read csv data, drop date coz not important
df_numpied = np.array(np.array_split((pd.read_csv("AUDUSD_20.11_1Y_1H.csv", usecols=[1,2,3,4])).to_numpy(),418))# uses only the middle colums and gets split into subsets of 20 (8360/20 = 418)

train=[]
test=[]

for i in range(0,len(df_numpied)):#  the shape 20 arrays into two, normalize and append to array.
    a,b=np.array_split(df_numpied[i],2)
    b=np.reshape(b,(1,40))# reshapes the current array to be a matrix(4x10) instead of a collection of vectors
    a=np.reshape(a,(1,40))

    train.append(tf.keras.utils.normalize(a))# pushes the normalized current array

    test.append(np.array([b[0][0],max(b[0]),min(b[0])]))# pushes the first, the max and the min of the current array

#train=tf.constant(train)# tensorfy the training data
#test=tf.constant(test)

def show_me_the_money(x):
    plt.plot([0,10],[test[x][0],test[x][1]],color='green')
    plt.plot([0,10],[test[x][0],test[x][2]],color='red')

    plt.text(10, test[x][1], str(test[x][1]), color='green')
    plt.text(10, test[x][2], str(test[x][2]),color='red')
    plt.text(0, test[x][0], str(test[x][0]), color='gray')
    #plt.text()
    plt.show()

show_me_the_money(21)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(40,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tf.keras.utils.plot_model(model, to_file='tmp/NN_1_visual.png', show_shapes=True)# Draw model
print(model.summary())

print("Fit model on training data")
history = model.fit(
    train,
    train,
    batch_size=20,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(test, test),)




"""
def prep_data(input,cut):
    slicer = input/cut
    df_numpied_f = np.array(np.array_split((pd.read_csv(input, usecols=[1, 2, 3, 4])).to_numpy(),slicer))
    train_f = []
    test_f = []
    for i in range(0, len(df_numpied_f)):
        a, b = np.array_split(df_numpied_f[i], 2)
        b = np.reshape(b, (1, 40))
        train_f.append(tf.keras.utils.normalize(a))
        test_f.append(np.array([b[0][0], max(b[0]), min(b[0])]))
    return [tf.constant(train_f),tf.constant(test_f)]
"""


#print(train[0])





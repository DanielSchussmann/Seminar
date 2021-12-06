import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# read csv data, drop date coz not important
#df_numpied = np.array(np.array_split((pd.read_csv("AUDUSD_20.11_1Y_1H.csv", usecols=[1,2,3,4],columns=["Open","High","Low","Close"])).to_numpy(),418))# uses only the middle colums and gets split into subsets of 20 (8360/20 = 418)
close = pd.read_csv("AUDUSD_20.11_1Y_1H.csv", usecols=[4])
lbls = close.columns[0]
df = close.copy()
df=np.array(np.array_split(df,836)).reshape([836,10])



x_train=df[::2]
y_train=np.zeros([len(x_train),len(x_train[0])])
x_val=df[1::2]#all the even indicies
y_val=np.zeros([len(x_val),len(x_val[0])])
print(x_train[0])
print(y_train[0])

plt.plot(df)
plt.show()

"""

train=[]
test=[]
for i in range(0,len(df_numpied)):#  the shape 20 arrays into two, normalize and append to array.
    a,b=np.array_split(df_numpied[i],2)
    b=b.reshape(1,40).astype("float32")# reshapes the current array to be a matrix(4x10) instead of a collection of vectors
    a=a.reshape(1,40).astype("float32")
    train.append(tf.keras.utils.normalize(a))# pushes the normalized current array
    test.append(np.array([b[0][0],max(b[0]),min(b[0])]))# pushes the first, the max and the min of the current array






train=np.reshape(train,(len(train),40)).astype("float32")

x_train=tf.constant(train)# tensorfy the training data
y_train=np.zeros([len(x_train),1]).astype("float32")#.reshape((-1, 1))

x_val=tf.constant(test)
y_val=np.zeros([len(x_val),1]).astype("float32")
"""







inputs = tf.keras.Input(shape=(10,), name="digits")
x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
tf.keras.utils.plot_model(model, to_file='tmp/NN_1_visual.png', show_shapes=True)# Draw model
print(model.summary())

model.fit(
    x_train,y_train,
    batch_size=20,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val,y_val))

accuracy = model.evaluate(x_train, x_val,verbose=0)
print(accuracy)























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
"""def show_me_the_money(x):
    plt.plot([0,10],[test[x][0],test[x][1]],color='green')
    plt.plot([0,10],[test[x][0],test[x][2]],color='red')
    plt.text(10, test[x][1], str(test[x][1]), color='green')
    plt.text(10, test[x][2], str(test[x][2]),color='red')
    plt.text(0, test[x][0], str(test[x][0]), color='gray')
    #plt.text()
    plt.show()
show_me_the_money(25)
"""
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(40,)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(3,activation ="relu"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build()
"""






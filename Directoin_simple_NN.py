import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt



data_training="market_data/AUD_USD.csv"
data_testing="market_data/AUD_CHF.csv"

#df_numpied = np.array(np.array_split((pd.read_csv("AUDUSD_20.11_1Y_1H.csv", usecols=[1,2,3,4],columns=["Open","High","Low","Close"])).to_numpy(),418))# uses only the middle colums and gets split into subsets of 20 (8360/20 = 418)
def OneD_data(inputfile):
    close = pd.read_csv(inputfile, usecols=[4])
    df = close.copy()

    df=np.array(np.array_split(df,int(len(df)/10))).reshape([int(len(df)/10),10])#testing examples

    x_raw =df.copy()
    x_norm=df[::2]
    for i in range(0,len(x_norm)):
        x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm=[]
    for i in range(0,len(df[1::2])):
        if max(df[1::2][i]) - df[1::2][i][0] > df[1::2][i][0] - min(df[1::2][i]):
            y_norm.append(1)
        else:
            y_norm.append(0)

    return [x_norm,y_norm,x_raw]


x_example=OneD_data(data_training)[0]
y_example=OneD_data(data_training)[1]

x_train,x_val = np.split(np.array(x_example),2)
y_train,y_val = np.split(np.array(y_example),2)

x_test=OneD_data(data_testing)[0]
y_test=OneD_data(data_testing)[1]
draw_test=OneD_data(data_testing)[2]




inputs = tf.keras.Input(shape=(10,), name="digits")
x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(1, activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.BinaryAccuracy(threshold=0.0,name="accuracy")])
#tf.keras.utils.plot_model(model, to_file='tmp/NN_1_visual.png', show_shapes=True)# Draw model
print(model.summary())

history= model.fit(
    x_train,y_train,
    batch_size=20,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val,y_val))


accuracy = lambda x,y: model.evaluate(x_test[x:y],np.array(y_test[x:y]))
prediction=lambda x,y: np.array(model.predict(x_test[x:y]))



print(str(accuracy(100,250)[0]*100)+'%')
hist_dick=history.history
plt.title('Loss')
plt.plot(hist_dick['loss'],color="coral")
plt.show()

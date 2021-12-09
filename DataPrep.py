import numpy as np
import pandas as pd
import tensorflow as tf




def OneD_data(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

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

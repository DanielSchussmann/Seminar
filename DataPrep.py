import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def OneD_data(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df=np.array(np.array_split(df,int(len(df)/10))).reshape([int(len(df)/10),10])#testing examples

    x_raw =df.copy()
    x_norm=df[::2]
    #for i in range(0,len(x_norm)):
     #   x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm=[]
    for i in range(0,len(df[1::2])):
        if max(df[1::2][i]) - df[1::2][i][0] > df[1::2][i][0] - min(df[1::2][i]): #if the max is bigger than the // change is the same still needs implementation
            y_norm.append(1)
        else:
            y_norm.append(-1)

    return [x_norm,y_norm,x_raw]



"""def mean_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples

    x_raw = df.copy()
    x_norm = df[::2]
    for i in range(0,len(x_norm)):
       x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(df[1::2][i]))

    return [x_norm, y_norm, x_raw]"""

def normalize(inp):
    A = [i*i for i in inp]
    v = (sum(A))**(0.5)
    A = [i / v for i in inp]
    return A

def mean_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples
    x_raw = df.copy()
    for i in range(0, len(df)):
        df[i]=normalize(df[i])

    x_norm = df[::2]
    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(df[1::2][i]))

    return [x_norm, y_norm, df,x_raw]



print(mean_movement('market_data/AUD_CHF.csv',[4]))



















def angular_price_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df=np.array(np.array_split(df,int(len(df)/10))).reshape([int(len(df)/10),10])#testing examples

    x_raw =df.copy()
    x_norm=df[::2]
    #for i in range(0,len(x_norm)):
     #   x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm=[]
    for i in range(0, len(df[1::2]) ):
        dom_change=max(max(df[1::2][i])-df[1::2][i][0] , df[1::2][i][0]-min(df[1::2][i]))
        y_norm.append(math.asin( dom_change / abs(df[1::2][i][0] - dom_change) ) )

    return [x_norm,y_norm,x_raw]


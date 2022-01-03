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

def relationfy(x):
    output = []
    no_order=[]
    #viz=[]
    for j in range(len(x)-1):
        temp=[]
        #color=(np.random.random(), np.random.random(), np.random.random())
        for i in range(j+1,len(x)):
            temp.append(x[j]/x[i])
            no_order.append(x[j]/x[i])
            #viz.append([[j,i],[x[j],x[i]]])
        output.append(temp)
    return [output,normalize(no_order)]


def mean_mvmnt(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples

    x_norm = []
    for i in range(0, len(df[::2])):
        x_norm.append(relationfy(df[::2][i])[1])

    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(normalize(df[1::2][i])))

    return [x_norm, y_norm, df]

#print(mean_mvmnt('market_data/AUD_CHF.csv',[4])[0][0:10])









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



#print(mean_movement('market_data/AUD_CHF.csv',[4]))



#df.to_csv('market_data/EUR_USD_D_corrected.csv')













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


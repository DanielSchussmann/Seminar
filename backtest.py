import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from DataPrep import *
from Visualize import draw_dis
inputfile='market_data/AUD_USD.csv'




data=OneD_data(inputfile,[4])[2]

portfolio=10000
p_start=portfolio
order_size=1000
account_risk=0.0009
nn = tf.keras.models.load_model('Neural_Networks/close_10_32_32_1')

"""
tp_buy=c_price*(1+account_risk)
sl_buy=c_price*(1-account_risk*0.5)

tp_sell=c_price*(1-account_risk)
sl_sell=c_price*(1+account_risk*0.5)
print()
"""
prediction=int(np.array(nn(data[3:4]))[0][0])
print(data[0:1])

print(prediction)



fail_checking=[]
#up_down prediction backtesting suit.
for j in range(0,10):
    current_batch=data[j]
    nn_data=data[j:j+1]
    price_zero= current_batch[0]
    port = portfolio
    prediction=int(np.array(nn(nn_data))[0][0])#np.random.randint(2)


    if prediction==0: #sell order
        tp = price_zero*(1-account_risk)
        sl = price_zero*(1+account_risk*0.5)
        for i in range(1,len(current_batch)): #go through values and check if they intersect with tp or sl
            if current_batch[i]<=tp:
                portfolio+= (tp-current_batch[0]) * order_size
                fail_checking.append([j,0,'tp'])
                break # the break emulates the order being sold

            if current_batch[i]>=sl:
                portfolio -= (current_batch[0]-sl) * order_size
                fail_checking.append([j, 0, 'sl'])
                break
        if port == portfolio:
            portfolio+=(current_batch[0]-current_batch[-1])* order_size
        draw_dis(current_batch, j, tp, sl, prediction[0][0], (-port + portfolio))





    else: #buy order
        tp = price_zero*(1+account_risk)
        sl = price_zero*(1-account_risk*0.5)
        for i in range(1,len(current_batch)):
            if current_batch[i]>=tp:
                portfolio+= (tp-current_batch[0]) * order_size
                fail_checking.append([j, 1, 'tp'])
                break
            if current_batch[i]<=sl:
                portfolio-= (current_batch[0]-sl) * order_size
                fail_checking.append([j, 1, 'sl'])
                break
        if port == portfolio:
            portfolio+=(current_batch[-1]-current_batch[0])* order_size
        draw_dis(current_batch,j, tp, sl,prediction,(-port+portfolio))






print(fail_checking)


print(p_start,portfolio)



#draw_dis(data[look],tp_buy,sl_buy)














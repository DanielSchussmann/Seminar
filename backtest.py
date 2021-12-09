import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DataPrep import *
from Visualize import draw_dis
inputfile='market_data/AUD_USD.csv'




data=OneD_data(inputfile,[4])[2]

portfolio=10000
order_size=1000
account_risk=0.0009

"""
tp_buy=c_price*(1+account_risk)
sl_buy=c_price*(1-account_risk*0.5)

tp_sell=c_price*(1-account_risk)
sl_sell=c_price*(1+account_risk*0.5)
print()
"""

output=[]
for i in range(0,10):
    current_batch=data[i]
    price_zero= current_batch[0]
    port = portfolio
    prediction=np.random.randint(2)

    if prediction==0: #sell order
        tp = price_zero*(1-account_risk)
        sl = price_zero*(1+account_risk*0.5)
        for i in range(0,len(current_batch)): #go through values and check if they intersect with tp or sl
            if current_batch[i]<tp:
                portfolio+= (tp-current_batch[0]) * order_size
                break # the break emulates the order being sold
            if current_batch[i]>sl:
                portfolio -= (current_batch[0]-sl) * order_size
                break
        if port == portfolio:
            portfolio+=(current_batch[0]-current_batch[-1])* order_size


    else: #buy order
        tp = price_zero*(1+account_risk)
        sl = price_zero*(1-account_risk*0.5)

        for i in range(0,len(current_batch)):
            if current_batch[i]>tp:
                portfolio+= (tp-current_batch[0]) * order_size
                break
            if current_batch[i]<sl:
                portfolio-= (current_batch[0]-sl) * order_size
                break

        if port == portfolio:
            portfolio+=(current_batch[-1]-current_batch[0])* order_size
        draw_dis(current_batch,i, tp, sl,prediction,(-port+portfolio))

    output.append(str(portfolio))




print(output)



#draw_dis(data[look],tp_buy,sl_buy)















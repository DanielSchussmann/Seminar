import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from DataPrep import *
from Visualize import draw_dis
import math
import json


def BT_directional_1(inputfile,model,portfolio_size,rng):
    portfolio = portfolio_size
    data=mean_movement(inputfile,[4])[2]
    data_raw=mean_movement(inputfile,[4])[3]
    leverage=300
    order_size=20*300#<-leverage

    nn = tf.keras.models.load_model(model) #the neural network to lead
    history=[]
    portfolio_development = [portfolio]

    for j in rng:
        port = portfolio
        A = [i * i for i in data_raw[j]]#denormalization of data
        deNorm = (sum(A)) ** (0.5)#denormalization of data

        #print(data_raw[j][0],data[j][0]*deNorm) -----> showcase normalization
        pre_batch = data[j]
        post_batch=data_raw[j+1]
        prediction=np.array(nn(data[j:j+1]))[0][0]
        pre_1 = pre_batch[-1]#last price from pre eavl batch
        risk_fctr=(prediction*deNorm*order_size*(max(pre_batch)/min(post_batch)))/(portfolio_size*100)

        if pre_1 < prediction:#------------->BUY

            tp = pre_1 * deNorm * (1+risk_fctr)
            sl = pre_1 * deNorm * (1-risk_fctr)

            pre_1 = pre_1 * deNorm
            prediction = prediction * deNorm

            for i in range(1, len(post_batch)):  # go through values and check if they intersect with tp or sl
                if post_batch[i] >= tp:
                    portfolio += (tp - pre_1) * order_size
                    history.append([j,'BUY/TP',pre_1,tp,sl,(tp - pre_1) * order_size])
                    portfolio_development.append(portfolio)
                    break  # the break emulates the order being sold
                elif post_batch[i] <= sl:
                    portfolio += (sl - pre_1) * order_size
                    history.append([j,'BUY/SL',pre_1,tp,sl,(sl - pre_1) * order_size])
                    portfolio_development.append(portfolio)
                    break
            if port == portfolio:
                portfolio += (post_batch[-1]-pre_1) * order_size
                history.append([j, 'BUY/NA', post_batch[-1], tp, sl, (post_batch[-1]-pre_1) * order_size])
                portfolio_development.append(portfolio)
            #draw_dis(post_batch, pre_1, j, tp, sl, 'BUY')


        elif pre_1>prediction:#------------->SELL

            tp = pre_1 * deNorm * (1 - risk_fctr)
            sl = pre_1 * deNorm * (1 + risk_fctr)

            pre_1 = pre_1 * deNorm
            prediction = prediction * deNorm
            for i in range(1, len(post_batch)):
                if post_batch[i] <= tp: #profit Trigger
                    portfolio += (pre_1-tp) * order_size
                    history.append([j,'SELL/TP',pre_1,tp,sl,(pre_1-tp) * order_size])
                    portfolio_development.append(portfolio)
                    break
                elif post_batch[i] >= sl:
                    portfolio += (pre_1-sl) * order_size
                    history.append([j, 'SELL/SL', pre_1, tp, sl, (pre_1-sl) * order_size])
                    portfolio_development.append(portfolio)
                    break
            if port == portfolio:
                portfolio += (pre_1-post_batch[-1]) * order_size
                history.append([j, 'SELL/NA', post_batch[-1], tp, sl, (pre_1-post_batch[-1]) * order_size])
                portfolio_development.append(portfolio)
            #draw_dis(post_batch,pre_1, j, tp, sl, 'SELL')
        else: #------------->NOTHING HAPPEND
            history.append(["bullshiiiiit"])

    profit=portfolio-portfolio_size
    return[portfolio_development,profit,np.array(history)]
    #prediction = int(np.array(nn(nn_data))[0][0])  # np.random.randint(2)

egg=BT_directional_1('market_data/AUD_CHF.csv','Neural_Networks/mean_direction',10000,range(0,150))

print(egg[0][-1])
fig,figr = plt.subplots()
figr.plot(egg[0])
figr.set_title('Portfolio development AUD/CHF')
fig.savefig('tmp/backtesting_plots/portfolio_AUD_CHF.svg')


"""
#up_down prediction backtesting suit.
for j in range(0,40):
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
        draw_dis(current_batch, j, tp, sl, prediction, (-port + portfolio))



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

history['orders'].append({
                        'Index': j,
                        'Type': 'SELL',
                        'Price': pre_1,
                        'Trigger': 'Stop Loss',
                        'loss': (sl-pre_1) * order_size})
"""












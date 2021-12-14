import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from DataPrep import *
from Visualize import draw_dis
import plotly.graph_objects as go
import math
import json


class BACKTEST():
    def __init__(self): #Innitializing all the required variables
        self.analytics=[]
        self.portfolio = 10000
        self.order_size=200
        self.leverage=50
        self.index=0
        self.order_id=1
        self.data=0
        self.open_orders=[]
        self.ohlc:np.array #OPENHIGHLOWCLOSE
        self.break_index=15

    def tick(self): #tick reffers to receiving a new candle from the market
        if len(self.open_orders) >= 1:
            self._TrackOrders()
        self.index+=1
        self.ohlc=self.data[self.index]

    def MakeOrder(self,type,tp,sl):
        buy_price=self.ohlc[-1]
        #"activating" order
        break_id=0
        order_id=self.index
        self.open_orders.append([type,tp,sl,buy_price,order_id,break_id])


    def _TrackOrders(self):
        for j in range(len(self.open_orders)):
            current_order=self.open_orders[j]
            type=current_order[0]
            tp=current_order[1]
            sl=current_order[2]
            buy_price=current_order[3]
            id=current_order[4]

            if type=='BUY':
                #Check wether or not the order is closed by tp or sl
                if tp <= self.ohlc[1]:# if tp is smaller than high
                    self.portfolio+=abs(tp-buy_price)*self.order_size*self.leverage
                    self.open_orders.pop(j) #close order
                    self.analytics.append([id,self.index,type+'/TP',str(self.ohlc),tp,sl,self.portfolio])
                    break
                elif sl>=self.ohlc[2]:# if sl is greater than low
                    self.portfolio -= abs(sl - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'/SL',str(self.ohlc),tp,sl,self.portfolio])
                    break #break the loop to stop index from becoming out of range
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += (self.ohlc[3] - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'TIMED_OUT', str(self.ohlc), tp, sl, self.portfolio])
                    break

            elif type=='SELL':
                if tp >= self.ohlc[2]:  # if tp is greater than low
                    self.portfolio += abs(tp - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)  # close order
                    self.analytics.append([id,self.index,type+'/TP',str(self.ohlc),tp,sl,self.portfolio])
                    break
                elif sl <= self.ohlc[1]:  # if sl is smaller than high
                    self.portfolio -= abs(sl - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'/SL',str(self.ohlc),tp,sl,self.portfolio])
                    break
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += (buy_price-self.ohlc[3]) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'TIMED_OUT', str(self.ohlc), tp, sl, self.portfolio])
                    break
            self.open_orders[j][5]+=1

    def visiulaize(self):
        fig = go.Figure(data=[go.Candlestick(
            open=self.data['Open'][:self.index],
            high=self.data['High'][:self.index],
            low=self.data['Low'][:self.index],
            close=self.data['Close'][:self.index], name='price')])
        count=1
        #fig.add_trace(go.Scatter(
        #    x=np.arange(0, len(close)), y=close, line=dict(color='blue', width=2), name='order'))
        for j in range(0,self.index):

            fig.add_shape(type="rect",
                      xref="x", yref="y",
                      x0=self.analytics[j][0], y0=self.analytics[j][4],
                      x1=self.analytics[j][1], y1=self.analytics[j][5],
                      line=dict(color="RoyalBlue", width=1),
                      fillcolor="LightSkyBlue",
                      opacity=0.5)
            fig.add_trace(go.Scatter(
            x=[np.mean(self.analytics[j][0:1])],
            y=[self.analytics[j][1] + 0.002],
            text=["order_{}".format(count)],
            mode="text"))
            count+=1
        fig.show()



        """viz_data = np.transpose(self.analytics)
        o_type, count = np.unique(viz_data[2], return_counts=True)
        portfolio = viz_data[-1].astype('float32')
        print(max(portfolio))
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(10)
        fig.set_figwidth(20)
        axs[0, 0].plot(viz_data[-1].astype('float32'))
        axs[0, 0].set_title('Portfolio')
        axs[0, 1].bar(o_type, count, color='coral')
        # axs[0, 1].bar(o_type,count, bottom=o_tpye[:1], color='y')
        axs[0, 1].set_title('Order_count')
        axs[1, 0].bar(['High/low', 'average'], [max(portfolio), np.mean(portfolio)], color='purple')
        #axs[1, 0].bar(['High/low', 'average'], [min(portfolio)], bottom=color = 'purple')
        axs[1, 0].set_title('Extrem')
        #axs[1, 1].plot(self.data[0:self.index], 'tab:red')
        axs[1, 1].set_title('Axis [1, 1]')
        i=0
        j=0
        while i< self.index:
            if self.data[j][0]>self.data[j][3]:
                axs[1, 1].plot([i,i],[self.data[j][1], self.data[j][2]], color='red', linewidth=1,) #wick
                axs[1, 1].plot([i,i],[self.data[j][0], self.data[j][3]], color='red', linewidth=5,) #body
            else:
                axs[1, 1].plot([i, i], [self.data[j][1], self.data[j][2]], color='green', linewidth=1, )  # wick
                axs[1, 1].plot([i, i], [self.data[j][0], self.data[j][3]], color='green', linewidth=5, )  # body
            i+=1
            j+=1
        plt.show()"""



nn=BACKTEST()
nn.data=np.array(pd.read_csv('market_data/AUD_USD.csv',usecols=[1,2,3,4]))
nn.tick()
nn.leverage=500
for i in range(0,50):
    gathered_data=[]
    look=[]
    for i in range(0,10):
        gathered_data.append(nn.ohlc[3])
        nn.tick()


    normalized_data = normalize(gathered_data)
    prediction_data=[normalized_data,normalized_data]

    A = [i * i for i in gathered_data]  # denormalization of data
    deNorm = (sum(A)) ** (0.5)  # denormalization of data

    pred = tf.keras.models.load_model('Neural_Networks/mean_direction')
    prediction=np.array(pred(np.array(prediction_data[0:1])))[0][0]*deNorm


    richt_preis=nn.ohlc[3]

    risk=(prediction*nn.order_size*(max(gathered_data)/min(gathered_data)))/(nn.portfolio*100)

    if richt_preis< prediction:
        tp = richt_preis * (1 + risk)
        sl = richt_preis * (1 - risk)
        nn.MakeOrder('BUY',tp,sl)
    elif richt_preis>prediction:
        tp = richt_preis * (1 - risk)
        sl = richt_preis * (1 + risk)
        nn.MakeOrder('SELL',tp,sl )


print(nn.portfolio)
#print(np.transpose(nn.analytics)) #groups up all the vaules for evaluation
for i in range(0,len(nn.analytics)):
    print(nn.analytics[i])

nn.visiulaize()







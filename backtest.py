import pandas as pd
import numpy as np
import tensorflow as tf
from DataPrep import *
import plotly.graph_objects as go
import math
import dash
import copy


class BACKTEST():
    def __init__(self): #Innitializing all the required variables
        self.analytics=[]
        self.portfolio = 10000
        self.order_size=200
        self.leverage=50
        self.index=0
        self.order_id=1
        self.data=0
        self.draw_data=0
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
        exe_id=self.index
        self.open_orders.append([type,tp,sl,buy_price,exe_id,break_id])


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
                    self.analytics.append([id,self.index,type+'/TP',buy_price,tp,sl,self.portfolio])
                    break
                elif sl>=self.ohlc[2]:# if sl is greater than low
                    self.portfolio -= abs(sl - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'/SL',buy_price,tp,sl,self.portfolio])
                    break #break the loop to stop index from becoming out of range
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += (self.ohlc[3] - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'TIMED_OUT',buy_price, tp, sl, self.portfolio])
                    break

            elif type=='SELL':
                if tp >= self.ohlc[2]:  # if tp is greater than low
                    self.portfolio += abs(tp - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)  # close order
                    self.analytics.append([id,self.index,type+'/TP',buy_price,tp,sl,self.portfolio])
                    break
                elif sl <= self.ohlc[1]:  # if sl is smaller than high
                    self.portfolio -= abs(sl - buy_price) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'/SL',buy_price,tp,sl,self.portfolio])
                    break
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += (buy_price-self.ohlc[3]) * self.order_size * self.leverage
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+'TIMED_OUT', buy_price, tp, sl, self.portfolio])
                    break
            self.open_orders[j][5]+=1

    def visiulaize(self):
        app = dash.Dash(__name__)
        fig_1 = go.Figure(data=[go.Candlestick(
            open=self.draw_data['Open'][0:self.index],
            high=self.draw_data['High'][0:self.index],
            low=self.draw_data['Low'][0:self.index],
            close=self.draw_data['Close'][0:self.index], name='price')],layout_title_text='Full Trade overview')

        portfolio = np.transpose(self.analytics)[-1].astype('float32')
        fig_2 = go.Figure(data=[go.Scatter(x=np.arange(0, len(portfolio)), y=portfolio,)],layout_title_text='Portfolio Evolution')
        #fig.add_trace(go.Scatter(
        #    x=np.arange(0, len(close)), y=close, line=dict(color='blue', width=2), name='order'))
        for j in range(0,len(self.analytics)):
            orders=self.analytics[j]
            start=orders[0]
            end=orders[1]
            o_type=orders[2]
            buy_price=orders[3]
            tp=orders[4]
            sl=orders[5]


            fig_1 .add_shape(type="rect", #POSITIVE RANGE
                      xref="x", yref="y",
                      x0=start, y0=tp if o_type[0:3]=='BUY' else buy_price, # To minimize code length
                      x1=end, y1=buy_price if o_type[0:3]=='BUY' else tp,
                      line=dict(color="#22754B", width=1),
                      fillcolor="#63ED7C",
                      opacity=0.3)
            fig_1 .add_shape(type="rect", #NEGATIVE RANGE
                          xref="x", yref="y",
                          x0=start, y0=sl if o_type[0:3]=='BUY' else buy_price,
                          x1=end, y1=buy_price if o_type[0:3]=='BUY' else sl,
                          line=dict(color="#752222", width=1),
                          fillcolor="#ED6363",
                          opacity=0.3)
            fig_1 .add_shape(type="line",
                          xref="x", yref="y",
                          x0=start, y0=buy_price,
                          x1=end, y1=buy_price,
                          line=dict(color="#752222", width=1,dash="dashdot"),)
        app.layout = dash.html.Div(children=[
            dash.html.H1(children='Backtesting summary'),

            dash.html.Div(children='''
                A summary of the backtesting process.
            '''),
            dash.dcc.Graph(id='Full order overview',
                      figure=fig_1 ),

        dash.dcc.Graph(id='Portfolio development',
                       figure=fig_2),
        ])

        if __name__ == '__main__':
            app.run_server(debug=True)




nn=BACKTEST()
nn.data=np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4]))
nn.tick()
nn.leverage=500
nn.break_index=5
for j in range(0,10):
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

    risk=10*(prediction*nn.order_size*(max(gathered_data)/min(gathered_data)))/(nn.portfolio*100)

    if richt_preis< prediction:
        tp = richt_preis * (1 + risk)
        sl = richt_preis * (1 - risk/2)
        nn.MakeOrder('BUY',tp,sl)
    elif richt_preis>prediction:
        tp = richt_preis * (1 - risk)
        sl = richt_preis * (1 + risk/2)
        nn.MakeOrder('SELL',tp,sl)


print(nn.portfolio)
#print(np.transpose(nn.analytics)) #groups up all the vaules for evaluation
for i in range(0,len(nn.analytics)):
    print(nn.analytics[i])
nn.draw_data=pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4])
nn.visiulaize()







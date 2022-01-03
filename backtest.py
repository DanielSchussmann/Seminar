import pandas as pd
import numpy as np
import tensorflow as tf
import uuid

from DataPrep import *
import plotly.graph_objects as go
import dash
import plotly
import sys
print(plotly.__version__)
volume_ref=np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[5]))
candles = np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4])[0:10])
"""
def pressure(refrence_volume,current_volume,candles):
    pressure_level=0
    candle_var=[abs(candles[i][0]-candles[i][4]) for i in range(len(candles))]
    avg_volume = np.mean(refrence_volume)
    check_volume=np.mean(current_volume)
    if check_volume > avg_volume:
        "egg"
    for i in range(0,len(current_volume)):
        avg_volume[i]=avg_volume[i]*1/(1+i)

    return candle_var

x,y=50,100
print(pressure(volume_ref[x:y],volume_ref[y:y+10],candles))

"""
print(sys.version)
class BACKTEST():
    def __init__(self): #Innitializing all the required variables
        self.analytics=[]
        self.portfolio = 10000
        self.order_size=200
        self.leverage=50
        self.index=0
        self.data=0
        self.draw_data=0
        self.open_orders=[]
        self.ohlc:np.array #OPENHIGHLOWCLOSE
        self.break_index=15


    def tick(self): #tick reffers to receiving a new candle from the market
        self.index+=1
        self.ohlc=self.data[self.index]
        self.volume = self.ohlc[4]

        if len(self.open_orders) >= 1:
            self._TrackOrders()

    def MakeOrder(self,type,tp,sl):
        buy_price=self.ohlc[3]
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
                    self.portfolio+= round(abs(tp-buy_price)*self.order_size*self.leverage,2)
                    self.open_orders.pop(j) #close order
                    self.analytics.append([id,self.index,type+' / TP',buy_price,tp,sl,self.portfolio])
                    break
                elif sl>=self.ohlc[2]:# if sl is greater than low
                    self.portfolio -= round(abs(sl - buy_price) * self.order_size * self.leverage,2)
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+' / SL',buy_price,tp,sl,self.portfolio])
                    break #break the loop to stop index from becoming out of range
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += round((self.ohlc[3] - buy_price) * self.order_size * self.leverage,2)
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+' / TIMED_OUT',buy_price, tp, sl, self.portfolio])
                    break

            elif type=='SELL':
                if tp >= self.ohlc[2]:  # if tp is greater than low
                    self.portfolio += round(abs(tp - buy_price) * self.order_size * self.leverage,2)
                    self.open_orders.pop(j)  # close order
                    self.analytics.append([id,self.index,type+' / TP',buy_price,tp,sl,self.portfolio])
                    break
                elif sl <= self.ohlc[1]:  # if sl is smaller than high
                    self.portfolio -= round(abs(sl - buy_price) * self.order_size * self.leverage,2)
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+' / SL',buy_price,tp,sl,self.portfolio])
                    break
                elif self.open_orders[j][5]==self.break_index: #break order after 10 ticks
                    self.portfolio += round((buy_price-self.ohlc[3]) * self.order_size * self.leverage,2)
                    self.open_orders.pop(j)
                    self.analytics.append([id,self.index,type+' / TIMED_OUT', buy_price, tp, sl, self.portfolio])
                    break
            self.open_orders[j][5]+=1

    def visualize(self):
        app = dash.Dash(__name__)
        #Figure-1-----------------------------------------------------------------------------------------------------------
        fig_1 = go.Figure(data=[go.Candlestick(increasing_line_color= 'rgba(44, 104, 82, 1)', increasing_fillcolor='rgba(44, 104, 82, 1)', decreasing_line_color='rgba(115, 0, 0, 1)', decreasing_fillcolor='rgba(115, 0, 0, 1)',
            open=self.draw_data['Open'][0:self.index],
            high=self.draw_data['High'][0:self.index],
            low=self.draw_data['Low'][0:self.index],
            close=self.draw_data['Close'][0:self.index], name='price',opacity=0.8)],layout_title_text='Trades overview')
        #fig_1.update_traces(decreasing_fillcolor=, selector=dict(type='candlestick'))

        # Figure-2------------------------------------------------------------------------------------------------------------------
        statistics =np.transpose(self.analytics)

        portfolio = statistics[-1].astype('float32')
        fig_2 = go.Figure(data=[go.Scatter(x=np.arange(0, len(portfolio)), y=portfolio,)],layout_title_text='Portfolio Evolution')
        fig_2.update_layout(hovermode="x",hoverlabel=dict(bgcolor="#636EFA",font_color='white',font_size=16,font_family="Arial"),yaxis_tickprefix = '$', yaxis_tickformat = ',.2f')
        #Figure-3------------------------------------------------------------------------------------------------------------------------------------
        exe_type, count = np.unique(statistics[2], return_counts=True)
        print(exe_type)
        colors=['rgba(75, 76, 177, 1)','rgba(75, 76, 177, 0.75)','rgba(75, 76, 177, 6)','rgba(238, 133, 56, 1)','rgba(238, 133, 56, 0.75)','rgba(238, 133, 56, 0.6)']
        fig_3= go.Figure(data=[go.Pie(values=count,labels=exe_type,marker_colors = colors,hole = 0.4)],layout_title_text='Order type Distribution')
        fig_3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Arial"))

        for j in range(0,len(self.analytics)):
            orders=self.analytics[j]
            start=orders[0]
            end=orders[1]
            o_type=orders[2]
            buy_price=orders[3]
            tp=orders[4]
            sl=orders[5]

            fig_1.add_shape(type="rect",name="Take Profit", #POSITIVE RANGE
                        xref="x", yref="y",x0=start,x1=end,
                        y0=tp if o_type[0:3]=='BUY' else buy_price, # To minimize code length
                        y1=buy_price if o_type[0:3]=='BUY' else tp,
                        line=dict(color="rgba(95, 251, 142, 1)", width=0.3,dash="dashdot"),fillcolor='rgba(95, 251, 142, 0.13)')
            fig_1.add_shape(type="rect",name="Stop Loss", #negative RANGE
                        xref="x", yref="y", x0=start, x1=end,
                        y0=sl if o_type[0:3]=='BUY' else buy_price,
                        y1=buy_price if o_type[0:3]=='BUY' else sl,
                        line=dict(color="rgba(251, 95, 95, 1)", width=1,dash="dashdot"),fillcolor='rgba(251, 95, 95, 0.13)')

            fig_1.add_shape(type="line", xref="x", yref="y",x0=start, y0=buy_price,x1=end, y1=buy_price,line=dict(color="#752222", width=1,dash="dashdot"),)
            fig_1.add_annotation(name="BUY order"if o_type[0:3]=='BUY' else 'SELL order',
                                x=start,
                                y=buy_price,
                                text=str(start),clicktoshow='onoff',
                                showarrow=True,
                                arrowcolor='Royalblue' if o_type[0:3]=='BUY' else'orange',
                                arrowhead=1,
                                arrowsize=2,
                                font_color='white',
                                ax=0,

                                ay=tp,
                                ayref='y',
                                bgcolor='Royalblue'if o_type[0:3]=='BUY' else 'orange' )
        fig_1.update_layout(xaxis_rangeslider_visible=False)
        fig_1.update_annotations(hoverlabel=dict(bgcolor='Royalblue'),hovertext='name')

        app.layout = dash.html.Div(children=[

            dash.html.H1(children='BACKTESTING SUMMARY',style={'textAlign': 'center','margin-bottom':'10px'}),
            dash.dcc.Graph(id='F_O_V',figure=fig_1,style={'fontsize':'30px','width': '98vw', 'height': '85vh','margin-bottom':'5px'}),
            dash.dcc.Graph(id='P_E',figure=fig_2,style={ 'height': '85vh'}),
            dash.dcc.Graph(id='T_T',figure=fig_3),
        ])



        if __name__ == '__main__':
            app.run_server(debug=True)



"""" +--------EXAMPLE--------+
nn=BACKTEST()
nn.data=np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4,5]))
nn.tick()
nn.leverage=500
nn.break_index=10

volume_ref=np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[5]))

for j in range(0,40):
    gathered_data=[]
    volume=[]
    for i in range(0,10):
        gathered_data.append(nn.ohlc[3])
        volume.append(nn.volume)
        nn.tick()


    normalized_data = normalize(gathered_data)
    prediction_data=[normalized_data,normalized_data]

    A = [i * i for i in gathered_data]  # denormalization of data
    deNorm = (sum(A)) ** (0.5)  # denormalization of data

    pred = tf.keras.models.load_model('Neural_Networks/mean_direction')
    prediction=np.array(pred(np.array(prediction_data[0:1])))[0][0]*deNorm


    richt_preis=nn.ohlc[3]



    if richt_preis< prediction:
        tp = prediction
        sl = richt_preis*0.997
        nn.MakeOrder('BUY',tp,sl)
    elif richt_preis>prediction:
        tp = prediction
        sl = richt_preis*1.003
        nn.MakeOrder('SELL',tp,sl)
    print(nn.volume)

print(nn.portfolio)
#print(np.transpose(nn.analytics)) #groups up all the vaules for evaluation
for i in range(0,len(nn.analytics)):
    print(nn.analytics[i])
nn.draw_data=pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4])
nn.visualize()
"""





class BACKTEST_multi_symbol():
    def __init__(self): #Innitializing all the required variables
        self.analytics={'orders':[],'other things?':[]}
        #self.symbols=['EURAUD','EURCHF','EURGBP','EURJPY','EURUSD']
        self.portfolio = 10000
        self.order_size=200
        self.leverage=50
        self.index=0
        self.data={'EURAUD':np.array(pd.read_csv('EURmajors/EURAUD_H.csv',usecols=[1,2,3,4,5])),
                   'EURCHF':np.array(pd.read_csv('EURmajors/EURCHF_H.csv',usecols=[1,2,3,4,5])),
                   'EURGBP':np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[1,2,3,4,5])),
                   'EURJPY':np.array(pd.read_csv('EURmajors/EURJPY_H.csv',usecols=[1,2,3,4,5])),
                   'EURUSD':np.array(pd.read_csv('EURmajors/EURUSD_H.csv',usecols=[1,2,3,4,5])),
                   }
        self.ea=0
        self.ec=0
        self.eg=0
        self.ej=0
        self.eu=0
        self.draw_data=0
        self.open_orders={}
        self.break_index=15


    def tick(self): #tick reffers to receiving a new candle from the market
        self.index+=1
        self.ea = self.data['EURAUD'][self.index]
        self.ec = self.data['EURCHF'][self.index]
        self.eg = self.data['EURGBP'][self.index]
        self.ej = self.data['EURJPY'][self.index]
        self.eu = self.data['EURUSD'][self.index]


    def MakeOrder(self,symbol,type):
        if symbol =='EURAUD':
                buy_price = self.ea[3]
        elif symbol =='EURCHF':
                buy_price = self.ec[3]
        elif symbol =='EURGBP':
                buy_price = self.eg[3]
        elif symbol =='EURJPY':
                buy_price = self.ej[3]
        elif symbol =='EURUSD':
                buy_price = self.eu[3]
        else:
            raise ValueError('Symbol "{}" is not known to BACKTEST'.format(symbol))

        if type!='LONG' and type!='SHORT':
            raise ValueError('Market execution "{}" is not known to BACKTEST'.format(type))

        exe_id=self.index
        hax=str(uuid.uuid1().hex) #generates a random hash dependent on time. Chance to overlap if there are 100000+ hashes generated at the same time, won't happen so it's fine.
        self.open_orders[hax]=[symbol,type,buy_price,exe_id]
        #self.analytics[hax] = [symbol, type,buy_price, exe_id]
        return hax


    def SellOrder(self,hax):
        buy_price = self.open_orders[hax][2]
        symbol = self.open_orders[hax][0]
        type = self.open_orders[hax][1]
        exe_id = self.open_orders[hax][3]
        end_id = self.index
        if symbol == 'EURAUD':
            sell_price= self.ea[3]
        elif symbol == 'EURCHF':
            sell_price = self.ec[3]
        elif symbol == 'EURGBP':
            sell_price = self.eg[3]
        elif symbol == 'EURJPY':
            sell_price = self.ej[3]
        elif symbol == 'EURUSD':
            sell_price = self.eu[3]
        else:
            raise ValueError('How the hell did "{}" end up here??'.format(symbol))

        if type=='LONG':
            self.portfolio+=(sell_price-buy_price)*self.order_size*self.leverage

        else:
            self.portfolio += (buy_price-sell_price)*self.order_size*self.leverage

        self.analytics['orders'].append([exe_id, end_id, symbol, type, buy_price, sell_price, self.portfolio, hax])
        del self.open_orders[hax]

dis = BACKTEST_multi_symbol()
dis.tick()
order1 = dis.MakeOrder('EURAUD','LONG')
order2 = dis.MakeOrder('EURUSD','LONG')
dis.tick()
dis.tick()
dis.SellOrder(order2)

order3 = dis.MakeOrder('EURJPY','LONG')
print(dis.open_orders)
print(dis.analytics['orders'])

import pandas as pd
import numpy as np
import tensorflow as tf
import uuid
from dash import dash_table
from DataPrep import *
import plotly.graph_objects as go
import dash
import plotly
import sys
from strats import *

global_deli=3000

load_data={
        'EURAUD':pd.read_csv('..//EURmajors/EURAUD_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURCHF':pd.read_csv('..//EURmajors/EURCHF_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURGBP':pd.read_csv('..//EURmajors/EURGBP_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURJPY':pd.read_csv('..//EURmajors/EURJPY_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURUSD':pd.read_csv('..//EURmajors/EURUSD_H.csv',usecols=[1,2,3,4])[0:global_deli],}



class BACKTEST_v3():
    def __init__(self): #Innitializing all the required variables
        self.analytics={'orders':[],'portfolio_mvmt':[],'EURUSD':[],'EURAUD':[],'EURJPY':[],'EURGBP':[],'EURCHF':[],'reasons':[]}
        self.portfolio = 10000
        self.analytics['portfolio_mvmt'].append(self.portfolio)
        self.order_size=200
        self.leverage=50
        self.index=0
        self.data=load_data.copy()
        self.symbols=np.array(list(self.data.keys()))
        self.open_orders={}
        self.main_color='#30AC83'

    def tick(self): #tick reffers to receiving a new candle from the market
        self.index+=1

    def MakeOrder(self,symbol,o_type):

        if np.array(np.where(self.symbols==symbol)).size==0: #print(type(symbol),self.symbols,np.array(np.where(self.symbols==symbol)))
                raise ValueError('Symbol "{}" is not known to BACKTEST'.format(symbol))
        if o_type != 'LONG' and o_type != 'SHORT':
            raise ValueError('Market execution "{}" is not known to BACKTEST'.format(o_type))

        buy_price = np.array(self.data[self.symbols[np.where(self.symbols==symbol)][0]])[self.index][3] # self.symbols[np.where(self.symbols==symbol)][0] ~ 'EURUSD'
                                                                                                        # np.array(self.data['EURUSD'])[0][index][close]
        exe_id=self.index
        hax=str(uuid.uuid1().hex) #generates a random hash dependent on time. Chance to overlap if there are 100000+ hashes generated at the same time, won't happen so it's fine.
        self.open_orders[hax] = [hax, symbol, o_type, buy_price, exe_id]
        #self.analytics[hax] = [symbol, type,buy_price, exe_id]
        return hax


    def SellOrder(self,hax):
        buy_price = self.open_orders[hax][3]
        symbol = self.open_orders[hax][1]
        type = self.open_orders[hax][2]
        exe_id = self.open_orders[hax][4]
        end_id = self.index
        sell_price = np.array(self.data[self.symbols[np.where(self.symbols==symbol)][0]])[self.index][3] #

        if type == 'LONG':
            profit = (sell_price-buy_price) * self.order_size * self.leverage
            self.portfolio += (sell_price-buy_price) * self.order_size * self.leverage
            self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price,  profit])

        else:
            profit = (buy_price-sell_price) * self.order_size * self.leverage
            self.portfolio += profit
            self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, profit])
        self.analytics['portfolio_mvmt'].append(self.portfolio)

        self.analytics[symbol].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, profit])
        del self.open_orders[hax]

#VISUALIZATION OF THEM DATA TINGS
    def _annotations(self, X):
        return [dict(
                        name=self.analytics[X][i][0],
                        x=self.analytics[X][i][1],
                        y=self.analytics[X][i][5],
                        text="긴" if self.analytics[X][i][4]=='LONG' else '짧은',
                        clicktoshow='onoff',
                        showarrow=True,
                        arrowcolor='#057FA6' if self.analytics[X][i][4]=='LONG' else 'coral',
                        arrowhead=1,arrowwidth=2,
                        arrowsize=1,
                        opacity=0.4,
                        ax=0,
                        hovertext=str(self.analytics[X][i][1])+' / '+str(self.analytics[X][i][0]),
                        ay=30 if self.analytics[X][i][4]=='LONG' else -30 ,
                        font_color='white',
                        bgcolor='#057FA6' if self.analytics[X][i][4]=='LONG' else 'coral') for i in range(len(self.analytics[X]))]

    #def internal_plot(self,PLOT,symbol,plot_type):
     #   self.addon[symbol][plot_type].append(PLOT)



    def _candel_plot(self,X):
        if len(self.analytics[X]) != 0:
            return go.Candlestick(
                            increasing_line_color='rgba(44, 104, 82, 1)', increasing_fillcolor='rgba(44, 104, 82, 1)',
                            decreasing_line_color='rgba(115, 0, 0, 1)', decreasing_fillcolor='rgba(115, 0, 0, 1)',
                            open=self.data[X]['Open'],
                            high=self.data[X]['High'],
                            low=self.data[X]['Low'],
                            close=self.data[X]['Close'], name=X, opacity=0.8)
        else:
            return go.Candlestick(
                            increasing_line_color='#DAFFF3', increasing_fillcolor='#DAFFF3',
                            decreasing_line_color='#DAFFF3', decreasing_fillcolor='#DAFFF3',
                            open=self.data[X]['Open'],
                            high=self.data[X]['High'],
                            low=self.data[X]['Low'],
                            close=self.data[X]['Close'], name=X, opacity=0.8)

    def init_layout(self):
        self.fig_ALL = go.Figure(data=[self._candel_plot(self.symbols[i])for i in range(len(self.symbols))],layout_title_text='Full Chart and Transaction list')
        self.fig_ALL.update_layout(xaxis_rangeslider_visible=False,template='simple_white')

        self.fig_port = go.Figure(data=[go.Scatter(x=np.arange(0, len(self.analytics['portfolio_mvmt'])), y=self.analytics['portfolio_mvmt'])],layout_title_text='Portfolio Evolution')
        self.fig_port.update_layout(template='simple_white', hovermode="x",hoverlabel=dict(bgcolor="#636EFA", font_color='white', font_size=16,font_family="Arial"), yaxis_tickprefix='$', yaxis_tickformat=',.2f')

        self.fig_acc = go.Figure(data=[go.Pie(values=[np.count_nonzero(np.transpose(self.analytics['orders'])[-1].astype('float32') < 0 ),np.count_nonzero(np.transpose(self.analytics['orders'])[-1].astype('float32') > 0 )],labels=['Lossable','Profitable'], hole=0.4, marker_colors=['#4824BA', '#BB4C71', '#FFCFE1'])],layout_title_text='Trade Count')
        self.fig_acc.update_traces(hoverinfo='label+value', hoverlabel=dict(font_size=26, font_family="Arial"))
        self.fig_acc.update_layout(annotations=[dict(text=len(self.analytics['orders']), x=0.5, y=0.5, font_size=50, showarrow=False)])

    def layout_callback(self,symbol):
        self.fig_1 = go.Figure(data = [self._candel_plot(symbol)],layout_title_text='{} Chart and Transaction list'.format(symbol))
        self.fig_1.update_layout(xaxis_rangeslider_visible = False, template = 'simple_white', annotations = self._annotations(symbol))
        #print(symbol)

    def draw(self):
        app = dash.Dash(__name__)
        app.title = "Backtest"
        #fig_acc.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Arial"),color_discrete_sequence=plotly.colors.sequential.RdBu)#accuracy of the bot in total also in regards to single currency
        DropDown_options = [[{'label': self.symbols[i], 'value': self.symbols[i]}][0] for i in range(len(self.symbols))]
        DropDown_options.append({'label':'ALL','value':'ALL'})
        print(DropDown_options)
        app.layout = dash.html.Div([
                dash.html.H1(children='BACKTESTING SUMMARY', style={'textAlign': 'center', 'font-size':'40px','margin-bottom': '20px'}),

                dash.html.Div([
                        dash.html.Div([dash.dcc.Graph(id="portfolio",figure=self.fig_port, style={'height': '60vh','width':'60vw'})],
                            style={'display':'flex','flex-direction': 'column','justify-content':'flex-start','align-items':'space-around','box-shadow':'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px'}),
                        dash.dcc.Graph(id="accuracy", figure=self.fig_acc, style={'height': '60vh','padding':'0px','border-radius':'10px','box-shadow':'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px'}),],
                    id='top_row',style={'display':'flex','flex-direction': 'row','justify-content':'space-around','align-items':'center','margin-bottom': '25px',}),

                dash.html.Div([
                    dash.dcc.Graph(id='F_O_V', figure=self.fig_ALL,style={ 'width': '100%', 'height': '85vh'}),
                    dash_table.DataTable(
                        id='table',
                        columns=([{'name':'Hash', 'id':'Hash','type':'any'},
                                  {'name': 'StartId', 'id': 'StartId', 'type': 'any'},
                                  {'name':'CloseId', 'id':'CloseId','type':'any'},
                                  {'name': 'Symbol', 'id': 'Symbol', 'type': 'any'},
                                  {'name': 'Type', 'id': 'Type', 'type': 'any'},
                                  {'name':'BuyPrice', 'id':'BuyPrice','type':'any'},
                                  {'name':'SellPrice', 'id':'SellPrice','type':'any'},
                                  {'name':'Profit', 'id':'Profit','type':'any'}]),style_cell={'textAlign': 'left'},
                        data=pd.DataFrame(self.analytics['orders'],columns=['Hash','StartId','CloseId', 'Symbol', 'Type', 'BuyPrice', 'SellPrice', 'Profit']).to_dict('records'),
                        editable=False)],
                    style={'fontsize': '30px', 'margin-bottom': '5px', 'width': '95.5vw','box-shadow': 'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px', 'align-self': 'center'}),

                dash.dcc.Dropdown(id='options',
                                options=DropDown_options,
                                value='ALL',
                                style={'width': '20vw', 'position': 'fixed'})],
            style={'display':'flex','flex-direction': 'column','justify-content':'flex-start','align-items':'space-around','padding':'0px','margin':'0px'})


        @app.callback(dash.Output('F_O_V', 'figure'), dash.Output('table','data'),[dash.Input('options', 'value')])
        def update_figure(value):
            if value == 'ALL':
                return self.fig_ALL, pd.DataFrame(self.analytics['orders'], columns=['Hash','StartId','CloseId', 'Symbol', 'Type', 'BuyPrice', 'SellPrice', 'Profit']).to_dict('records')
            if value!='ALL':
                self.layout_callback(value)
                return self.fig_1, pd.DataFrame(self.analytics[value], columns=['Hash','StartId','CloseId', 'Symbol', 'Type', 'BuyPrice', 'SellPrice', 'Profit']).to_dict('records')

        if __name__ == '__main__':
                app.run_server(debug=True)



dis = BACKTEST_v3()


[dis.tick() for x in range(0,30)]
standard_deviation=lambda data: np.sum(((data-(np.mean(data)))**2)/len(data))**0.5
my_orders=[]


pred = tf.keras.models.load_model('..//Neural_Networks/promise_1')
for x in range(0,100):
    prediction_data=[]
    for f in range(len(dis.symbols)):
       prediction_data.append(np.array(normalize([ dis.data[dis.symbols[f]]['Close'][x]/dis.data[dis.symbols[f]]['Close'][dis.index] for x in range(dis.index - 10,dis.index)])))

    prediction_data=np.asarray(prediction_data)
    #print(prediction_data)
    #prediction = pred.pssredict_on_batch(prediction_data_1)
    prdctn= pred.predict_on_ssbatch(prediction_data)

    for t in range(len(dis.symbols)):
       #print(np.where(np.amax(prdctn[t]))[0][0])
       print(prdctn, np.amax(prdctn[t]))
       if np.where(np.amax(prdctn[t]))[0][0] == 1:
            continue
       elif np.where(np.amax(prdctn[t]))[0][0] == 0:
           xxx = dis.MakeOrder(dis.symbols[t], 'SHORT')
           my_orders.append(xxx)
       elif np.where(np.amax(prdctn[t]))[0][0] ==2:
           xxx = dis.MakeOrder(dis.symbols[t], 'SHORT')
           my_orders.append(xxx)


    o = 0
    while o<len(my_orders):
        #print(dis.open_orders[my_orders[o]])
        if dis.open_orders[my_orders[o]][4] + 5 < dis.index:
            dis.SellOrder(my_orders[o])
            my_orders.pop(o)
        o += 1
    dis.tick()
#print(dis.analytics)

dis.init_layout()
dis.draw()




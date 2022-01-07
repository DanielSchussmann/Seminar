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
        'EURAUD':pd.read_csv('EURmajors/EURAUD_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURCHF':pd.read_csv('EURmajors/EURCHF_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURGBP':pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURJPY':pd.read_csv('EURmajors/EURJPY_H.csv',usecols=[1,2,3,4])[0:global_deli],
        'EURUSD':pd.read_csv('EURmajors/EURUSD_H.csv',usecols=[1,2,3,4])[0:global_deli],}



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


eragon = BACKTEST_v3()
eragon.tick()
order = eragon.MakeOrder('EURUSD','SHORT')
eragon.tick()
order2 = eragon.MakeOrder('EURJPY','LONG')
eragon.tick()
eragon.SellOrder(order)
eragon.tick()
eragon.tick()
eragon.tick()
eragon.SellOrder(order2)
eragon.init_layout()

eragon.draw()

#####################################--------------------------------------------------------------------------------------













class BACKTEST_v2():
    def __init__(self): #Innitializing all the required variables
        self.analytics={'orders':[],'portfolio_mvmt':[],'EURUSD':[],'EURAUD':[],'EURJPY':[],'EURGBP':[],'EURCHF':[],'reasons':[]}
        self.portfolio = 10000
        self.analytics['portfolio_mvmt'].append(self.portfolio)
        self.order_size=200
        self.leverage=50
        self.index=0
        self.data=load_data.copy()
        self.ea = self.data['EURAUD']
        self.ec = self.data['EURCHF']
        self.eg = self.data['EURGBP']
        self.ej = self.data['EURJPY']
        self.eu = self.data['EURUSD']
        self.draw_data=0
        self.open_orders={}
        self.break_index=15


    def tick(self): #tick reffers to receiving a new candle from the market
        self.index+=1



    def MakeOrder(self,symbol,type):
        if symbol =='EURAUD':
                buy_price = np.array(self.ea)[self.index][3]
        elif symbol =='EURCHF':
                buy_price = np.array(self.ec)[self.index][3]
        elif symbol =='EURGBP':
                buy_price = np.array(self.eg)[self.index][3]
        elif symbol =='EURJPY':
                buy_price = np.array(self.ej)[self.index][3]
        elif symbol =='EURUSD':
                buy_price = np.array(self.eu)[self.index][3]
        else:
            raise ValueError('Symbol "{}" is not known to BACKTEST'.format(symbol))

        if type!='LONG' and type!='SHORT':
            raise ValueError('Market execution "{}" is not known to BACKTEST'.format(type))

        exe_id=self.index
        hax=str(uuid.uuid1().hex) #generates a random hash dependent on time. Chance to overlap if there are 100000+ hashes generated at the same time, won't happen so it's fine.
        self.open_orders[hax] = [hax, symbol, type, buy_price, exe_id]
        #self.analytics[hax] = [symbol, type,buy_price, exe_id]
        return hax


    def SellOrder(self,hax):
        buy_price = self.open_orders[hax][3]
        symbol = self.open_orders[hax][1]
        type = self.open_orders[hax][2]
        exe_id = self.open_orders[hax][4]
        end_id = self.index
        if symbol == 'EURAUD':
            sell_price= np.array(self.ea)[self.index][3]
        elif symbol == 'EURCHF':
            sell_price = np.array(self.ec)[self.index][3]
        elif symbol == 'EURGBP':
            sell_price = np.array(self.eg)[self.index][3]
        elif symbol == 'EURJPY':
            sell_price = np.array(self.ej)[self.index][3]
        elif symbol == 'EURUSD':
            sell_price = np.array(self.eu)[self.index][3]
        else:
            raise ValueError('How the hell did "{}" end up here??'.format(symbol))

        if type == 'LONG':
            profit = (sell_price-buy_price) * self.order_size * self.leverage
            self.portfolio += (sell_price-buy_price) * self.order_size * self.leverage
            self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, 'fail']) if profit < 0 else self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, 'succ'])

        else:
            profit = (buy_price-sell_price) * self.order_size * self.leverage
            self.portfolio += profit
            self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, 'fail']) if profit<0 else self.analytics['orders'].append([hax, exe_id, end_id, symbol, type, buy_price, sell_price, 'succ'])
        self.analytics['portfolio_mvmt'].append(self.portfolio)

        self.analytics[symbol].append([hax, exe_id, end_id, type, buy_price, sell_price])
        del self.open_orders[hax]



#####################################--------------------------------------------------------------------------------------

class VIZ():
    def __init__(self):
        self.data=load_data
        self.leng=1000
        self.main_color='#30AC83'
        self.order_history=0
        self.analytics={}
        self.addon= {'EURUSD': {'Shape':[],'Scatter':[]} ,'EURAUD':{'Shape':[],'Scatter':[]},'EURJPY':{'Shape':[],'Scatter':[]},'EURGBP':{'Shape':[],'Scatter':[]},'EURCHF':{'Shape':[],'Scatter':[]}}
    def annotations(self, X):
        return [dict(
                        name=self.analytics[X][i][0],
                        x=self.analytics[X][i][1],
                        y=self.analytics[X][i][4],
                        text="긴" if self.analytics[X][i][3]=='LONG' else '짧은',
                        clicktoshow='onoff',
                        showarrow=True,
                        arrowcolor='#057FA6' if self.analytics[X][i][3]=='LONG' else 'coral',
                        arrowhead=1,arrowwidth=2,
                        arrowsize=1,
                        opacity=0.4,
                        ax=0,
                        hovertext=str(self.analytics[X][i][1])+' / '+str(self.analytics[X][i][0]),
                        ay=30 if self.analytics[X][i][3]=='LONG' else -30 ,
                        font_color='white',
                        bgcolor='#057FA6' if self.analytics[X][i][3]=='LONG' else 'coral') for i in range(len(self.analytics[X]))]

    def internal_plot(self,PLOT,symbol,plot_type):
        self.addon[symbol][plot_type].append(PLOT)



    def candel_plot(self,X):
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
        self.fig_ALL = go.Figure(data=[self.candel_plot('EURUSD'),self.candel_plot('EURGBP'),self.candel_plot('EURJPY'),self.candel_plot('EURCHF'),self.candel_plot('EURAUD')],layout_title_text='Chart and Transaction list')
        self.fig_ALL.update_layout(xaxis_rangeslider_visible=False,template='simple_white')

        self.fig_EURUSD = go.Figure(data=[self.candel_plot('EURUSD')])
        self.fig_EURUSD.update_layout(xaxis_rangeslider_visible=False, template='simple_white', annotations=self.annotations('EURUSD'))

        #[fig_EURUSD.add_shape(self.addon['EURUSD']['Scatter'][an]) for an in range(len(self.addon['EURUSD']['Scatter']))]


        self.fig_EURGBP = go.Figure(data=[self.candel_plot('EURGBP')])
        self.fig_EURGBP.update_layout(xaxis_rangeslider_visible=False, template='simple_white', annotations=self.annotations('EURGBP'))

        self.fig_EURJPY = go.Figure(data=[self.candel_plot('EURJPY')])
        self.fig_EURJPY.update_layout(xaxis_rangeslider_visible=False, template='simple_white', annotations=self.annotations('EURJPY'))

        self.fig_EURCHF = go.Figure(data=[self.candel_plot('EURCHF')])
        self.fig_EURCHF.update_layout(xaxis_rangeslider_visible=False, template='simple_white', annotations=self.annotations('EURCHF'))

        self.fig_EURAUD = go.Figure(data=[self.candel_plot('EURAUD')])
        self.fig_EURAUD.update_layout(xaxis_rangeslider_visible=False, template='simple_white', annotations=self.annotations('EURAUD'))

        self.fig_port =go.Figure(data=[go.Scatter(x=np.arange(0, len(self.analytics['portfolio_mvmt'])), y=self.analytics['portfolio_mvmt'] )],layout_title_text='Portfolio Evolution')
        self.fig_port.update_layout(template='simple_white',hovermode="x",hoverlabel=dict(bgcolor="#636EFA", font_color='white', font_size=16, font_family="Arial"),yaxis_tickprefix = '$', yaxis_tickformat = ',.2f')

        self.fig_acc = go.Figure(data=[go.Pie(values=[np.count_nonzero(np.transpose(self.analytics['orders'])[-1]=='succ'), np.count_nonzero(np.transpose(self.analytics['orders'])[-1]=='fail'), 0], labels=['Succ', 'Fail', 'Random'], hole=0.4,marker_colors=['#4824BA','#BB4C71','#FFCFE1'])],layout_title_text='Trade Count')
        self.fig_acc.update_traces(hoverinfo='label+value',hoverlabel=dict(font_size=26, font_family="Arial"))
        self.fig_acc.update_layout(annotations=[dict(text=len(self.analytics['orders']), x=0.5, y=0.5, font_size=50, showarrow=False)])

    def draw(self):
        app = dash.Dash(__name__)
        app.title = "Backtest"
        #fig_acc.update_layout(hoverlabel=dict(bgcolor="white", font_size=16, font_family="Arial"),color_discrete_sequence=plotly.colors.sequential.RdBu)#accuracy of the bot in total also in regards to single currency
        app.layout = dash.html.Div([
                dash.html.H1(children='BACKTESTING SUMMARY', style={'textAlign': 'center', 'font-size':'40px','margin-bottom': '20px'}),

                dash.html.Div([
                    dash.html.Div([
                        #dash.html.H4(children='Portfolio Evolution',style={'textAlign': 'center', 'font-size': '25px', 'font-style':'italic','background-color':'red'}),
                        dash.dcc.Graph(id="portfolio",figure=self.fig_port, style={'height': '60vh','width':'60vw'}),
                    ], style={'display':'flex','flex-direction': 'column','justify-content':'flex-start','align-items':'space-around','box-shadow':'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px'}),
                    dash.dcc.Graph(id="accuracy", figure=self.fig_acc, style={'height': '60vh','padding':'0px','border-radius':'10px','box-shadow':'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px'}),
                ], id='top_row',style={'display':'flex','flex-direction': 'row','justify-content':'space-around','align-items':'center','margin-bottom': '25px',}),

                dash.html.Div([
                    dash.dcc.Graph(id='F_O_V', figure=self.fig_ALL,style={ 'width': '100%', 'height': '85vh'}),
                    dash_table.DataTable(
                    id='table',
                    columns=([{'name':'Hash', 'id':'Hash','type':'any'},
                              {'name': 'StartId', 'id': 'StartId', 'type': 'any'},
                              {'name':'CloseId', 'id':'CloseId','type':'any'},
                              {'name': 'Type', 'id': 'Type', 'type': 'any'},
                              {'name':'BuyPrice', 'id':'BuyPrice','type':'any'},
                              {'name':'SellPrice', 'id':'SellPrice','type':'any'}]),style_cell={'textAlign': 'left'},
                    data=pd.DataFrame(self.analytics['EURUSD'],columns=['Hash','StartId','Type', 'CloseId', 'BuyPrice', 'SellPrice']).to_dict('records'),
                    editable=False)
                ], style={'fontsize': '30px', 'margin-bottom': '5px', 'width': '95.5vw','box-shadow': 'rgba(100, 100, 111, 0.2) 0px 7px 29px 0px', 'align-self': 'center'}),
            dash.dcc.Dropdown(id='options',
                              options=[{'label': 'ALL', 'value': 'ALL'},
                                       {'label': 'EURUSD', 'value': 'EURUSD'},
                                       {'label': 'EURGBP', 'value': 'EURGBP'},
                                       {'label': 'EURJPY', 'value': 'EURJPY'},
                                       {'label': 'EURCHF', 'value': 'EURCHF'},
                                       {'label': 'EURAUD', 'value': 'EURAUD'},
                                       ], value='ALL', style={'width': '20vw', 'position': 'fixed'}
                              ),
        ],style={'display':'flex','flex-direction': 'column','justify-content':'flex-start','align-items':'space-around','padding':'0px','margin':'0px'})



        @app.callback(dash.Output('F_O_V', 'figure'), dash.Output('table','data'), [dash.Input('options', 'value')])
        def update_figure(value):
            if value == 'EURUSD':
                return self.fig_EURUSD, pd.DataFrame(self.analytics['EURUSD'], columns=['Hash', 'StartId', 'CloseId', 'Type','BuyPrice', 'SellPrice']).to_dict('records')
            if value == 'EURGBP':
                return self.fig_EURGBP, pd.DataFrame(self.analytics['EURGBP'], columns=['Hash', 'StartId', 'CloseId','Type', 'BuyPrice', 'SellPrice']).to_dict('records')
            if value == 'EURJPY':
                return self.fig_EURJPY, pd.DataFrame(self.analytics['EURJPY'], columns=['Hash', 'StartId', 'CloseId','Type', 'BuyPrice', 'SellPrice']).to_dict('records')
            if value == 'EURCHF':
                return self.fig_EURCHF, pd.DataFrame(self.analytics['EURCHF'], columns=['Hash', 'StartId', 'CloseId', 'Type','BuyPrice', 'SellPrice']).to_dict('records')
            if value == 'EURAUD':
                return self.fig_EURAUD, pd.DataFrame(self.analytics['EURAUD'], columns=['Hash', 'StartId', 'CloseId','Type', 'BuyPrice', 'SellPrice']).to_dict('records')
            if value == 'ALL':
                return self.fig_ALL, pd.DataFrame([self.analytics['EURUSD']], columns=['Hash', 'StartId', 'CloseId','Type', 'BuyPrice', 'SellPrice']).to_dict('records')


        if __name__ == '__main__':
                app.run_server(debug=True)


dis = BACKTEST_v2()
vis = VIZ()

[dis.tick() for x in range(0,30)]
standard_deviation=lambda data: np.sum(((data-(np.mean(data)))**2)/len(data))**0.5
my_orders=[]
symbols = [[dis.ea,'EURAUD'],[dis.eg,'EURGBP'],[dis.ej,'EURJPY'],[dis.eu,'EURUSD'],[dis.ec,'EURCHF']]


for j in range(30,global_deli-30):

    if j%20==0:

#Check for inn
        for s in range(0,len(symbols)):
            cur_candel = np.array(symbols[s][0])[dis.index]
            cur_close = cur_candel[3]
            rng = np.array(symbols[s][0])[dis.index-20:dis.index]
            cur_mean = np.mean(rng)
            cur_deviation = standard_deviation(rng)
            #print(rng,cur_deviation,cur_mean)
            #print(symbols[s][1],cur_candel,cur_deviation,cur_mean,dis.index)
            if  cur_candel.any() < cur_mean - cur_deviation:
                xxx = dis.MakeOrder(symbols[s][1], 'SHORT')
                my_orders.append(xxx)
            if cur_mean + cur_deviation < cur_candel.any():
                xxx = dis.MakeOrder(symbols[s][1], 'LONG')
                my_orders.append(xxx)


#check for exit condition
    o = 0
    while o<len(my_orders):
        #print(dis.open_orders[my_orders[o]])
        if dis.open_orders[my_orders[o]][4] + 5 < dis.index:
            dis.SellOrder(my_orders[o])
            my_orders.pop(o)
        o += 1
    dis.tick()



#vis = VIZ()
#vis.analytics = dis.analytics
#vis.order_history = pd.DataFrame.from_dict(dis.open_orders, orient='index', columns=['Hash', 'Symbol', 'OrderType', 'BuyPrice', 'Index'])
#vis.init_layout()
#is.internal_plot(go.Scatter(x=[1,2,3],y=[1,1,1]),'EURUSD','Shape')
#vis.draw()



























































"""
volume_ref=np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[5]))
candles = np.array(pd.read_csv('market_data/AUD_CHF.csv',usecols=[1,2,3,4])[0:10])

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

"""

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


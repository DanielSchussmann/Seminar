import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
app = dash.Dash(__name__)

data_read=pd.read_csv('market_data/AUD_USD.csv',usecols=[1,2,3,4])[50:600]
data=np.array(data_read['Close']).reshape((len(data_read)))

fig_1 = go.Figure(data=[go.Candlestick(
            x=data_read['Local Time'],
            open=data_read['Open'],
            high=data_read['High'],
            low=data_read['Low'],
            close=data_read['Close'], name='price',opacity=0.8)],layout_title_text='Trades overview')
EMA=data[0]


sma = lambda n,data:[np.sum(data[x-n:x])/n for x in range(len(data),n,-1)]

sma_14 = sma(14,data)
sma_21 = sma(21,data)
sma_28 = sma(28,data)
sma_40 = sma(40,data)
sma_50 = sma(50,data)
fig_1.add_trace(go.Scatter(x=np.arange(len(data),10,-1), y=sma_14, line=dict(color='#E1D623'),name="SMA_14"))
fig_1.add_trace(go.Scatter(x=np.arange(len(data),20,-1), y=sma_21, line=dict(color='#097363'),name="SMA_21"))
fig_1.add_trace(go.Scatter(x=np.arange(len(data),30,-1), y=sma_28, line=dict(color='#3A3CC3'),name="SMA_28"))
fig_1.add_trace(go.Scatter(x=np.arange(len(data),40,-1), y=sma_40, line=dict(color='#923AC3'),name="SMA_40"))
fig_1.add_trace(go.Scatter(x=np.arange(len(data),50,-1), y=sma_50, line=dict(color='#C33AAF'),name="SMA_50"))

fig_1.update_layout(xaxis_rangeslider_visible=False)
app.layout = dash.html.Div(children=[
    dash.html.H1(children='BACKTESTING SUMMARY', style={'textAlign': 'center', 'margin-bottom': '10px'}),
    dash.dcc.Graph(id='F_O_V', figure=fig_1, style={'fontsize': '30px', 'width': '98vw', 'height': '85vh', 'margin-bottom': '5px'}),])

#if __name__ == '__main__':
 #   app.run_server(debug=True)
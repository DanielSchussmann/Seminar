import numpy as np
import threading
import json
from datetime import datetime


order={}
order['orders']=[]

portfolio1=10000
trade_amount=10000/100

#ticking
def printit():
  threading.Timer(5.0, printit).start()



def orderToJson(symbol,current_price, top, bot):
    order['orders'].append({
    'symbol':symbol,
    'time': datetime.now().strftime("%H:%M"),
    'price': current_price,
    'top': top,
    'bot': bot})

    print(order['orders'][0])


printit()

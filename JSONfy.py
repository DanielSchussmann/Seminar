import numpy as np
import threading
import json
from datetime import datetime
import tensorflow as tf
import math
import matplotlib.pyplot as plt

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

x =[0.69814,0.69871,0.69998,0.70062,0.70125,0.70258,0.70362,0.70329,0.70375,0.70434,0.70547,0.70525,0.70525,0.70537,0.70503]

def normalize(inp):
    A = [i*i for i in inp]
    v = (sum(A))**(0.5)
    A = [i / v for i in inp]
    return A
x_norm= normalize(x)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x,color="purple")
ax1.set_title('Standard Data', fontsize=9,pad=-14)
ax2.plot(x_norm,color="cyan")
ax2.set_title('Normalized Data', fontsize=9,pad=-14)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter
all = pd.read_csv("market_data/AUD_USD.csv", usecols=[1, 2, 3, 4])
dt= all.to_numpy().copy()
dt =dt[50:-1]


matplotlib.style.use('bmh')
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim(0.001,50)
ax.set_ylim(0.725,0.75)
ax.set_xlabel('time')
ax.set_ylabel('Price')
ax.set_title('AI-visualization')
short_c = '#D13838'
long_c = '#63ED7C'


def animate(i=30):
    if i%11==0:
        ax.plot([i,i+4],[dt[i][0]-0.001,dt[i][0]-0.001],color="coral",linewidth=1,linestyle="dashed")
        ax.plot([i, i + 4], [dt[i][3]+0.001, dt[i][3]+0.001], color="cyan",linewidth=1,linestyle="dashed")
    if dt[i][0]>dt[i][3]:
        ax.plot([i,i], [dt[i][1], dt[i][2]], color=long_c, linewidth=1,)
        ax.plot([i, i], [dt[i][0], dt[i][3]], color=long_c, linewidth=5, )
    else:
        ax.plot([i, i], [dt[i][1], dt[i][2]], color=short_c, linewidth=1, )
        ax.plot([i, i], [dt[i][0], dt[i][3]], color=short_c, linewidth=5, )

anim = FuncAnimation(fig, animate, interval=20000,frames=50)
anim.save("tmp/movie.gif", writer=PillowWriter(fps=1))








def draw_candles(data):
    short_c='#D13838'
    long_c='#63ED7C'
    matplotlib.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    i=0
    j=0
    while i<len(data):
        if data[j][0]>data[j][3]:
            ax.plot([i,i],[data[j][1], data[j][2]], color=short_c, linewidth=1,) #wick
            ax.plot([i,i],[data[j][0], data[j][3]], color=short_c, linewidth=5,) #body
        else:
            ax.plot([i, i], [data[j][1], data[j][2]], color=long_c, linewidth=1, )  # wick
            ax.plot([i, i], [data[j][0], data[j][3]], color=long_c, linewidth=5, )  # body
        i+=1
        j+=1

    ax.margins(x=0)
    fig.set_size_inches(len(data)*0.2, len(data)*0.1)
    #plt.show()
    plt.savefig("tmp/test_1.png",bbox_inches='tight')
#draw_candles(dt[100:210])



#['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

def draw_dis(data,tp,sl):
    matplotlib.style.use('bmh')
    fig, ax = plt.subplots()
    ax.axhline(y=tp,color='g',label='Take-Profit',linewidth=1,linestyle='dashed')
    ax.axhline(y=sl,color='r',label="Stop-Soss",linewidth=1,linestyle='dashed')
    ax.axhline(y=data[0][0], color='0',linestyle='dashed',label="Starting price",linewidth=1)
    ax.plot(data,label="Price move",color="coral")
    rect = matplotlib.patches.Rectangle((1,data[0][0]), 10,tp-data[0][3],linestyle='dashed', alpha=0.5,linewidth=1, edgecolor='g', facecolor='g')

    ax.legend()
    ax.add_patch(rect)
    plt.show()


from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

_, stego = wavfile.read('../animation_stego_music/stego_full.wav')

fe_c = 48000
fact = 30
xmin = 0
xmax = 3
nbx = 100
x = np.linspace(xmin, xmax, nbx)

fig, ax  = plt.subplots(1,1,figsize=(5,3))
line, = ax.plot([],[],linewidth=0.1) 
ax.set_xlim(-1,len(stego)//fe_c+1)
ax.set_ylim(-15000,15000)
ax.grid()
ax.set_xlabel(r'$t$ (sec)')
ax.yaxis.set_ticklabels([])
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

n = 30
delta_t1, delta_t2 = len(stego)/fe_c, len(stego)/(fact*fe_c)
def animate(i): 
    t = np.linspace(0, (delta_t1-delta_t2)*(n-i)/n, len(stego))
    line.set_data(t, stego)
    
    fig2, ax2  = plt.subplots(1,1,figsize=(5,3))
    ax2.plot(t, stego,linewidth=0.1, color='#90be6d') 
    ax2.set_xlim(-1,len(stego)//fe_c+1)
    ax2.set_ylim(-15000,15000)
    ax2.grid()
    ax2.set_xlabel(r'$t$ (sec)')
    ax2.yaxis.set_ticklabels([])
    ## Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.savefig('../animation_stego_music/'+str(i)+'.png', format='png',dpi=300)
    
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n, blit=True, interval=15, repeat=False)
plt.show()

rc('animation', html='jshtml')
ani
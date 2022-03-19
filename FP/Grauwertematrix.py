# Marius Wenk, Fernando Grumpe

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits import mplot3d


w=640
h=480

im_arr=[]
px_arr=[]

# for i in range(37):
#     im_arr.append(Image.open("/home/fernando/Desktop/FP/laser/bilder/Bild "+str(i+610)+".jpg"))
#     px_arr.append(im_arr[i].load())

indizes = [21,22,23,24,25,26,27,30,11,12,13,14]

pic_0 = []
px_0 = []
px_r = []
px_g = []
px_b = []
px_gray = []
count = 0
for i in indizes:
    pic_0.append(Image.open(f"4.Daten/bilder/Bild {i:03d}.jpg"))
    px_0.append(pic_0[count].load())
    px_r.append(np.zeros(shape=(w,h)))
    px_g.append(np.zeros(shape=(w,h)))
    px_b.append(np.zeros(shape=(w,h)))
    px_gray.append(np.zeros(shape=(w,h)))
    count += 1

for n in range(len(indizes)): 
    for i in range(w):
    	for j in range(h):
    		px_r[n][i,j]= px_0[n][(i,j)][0]
    		px_g[n][i,j]= px_0[n][(i,j)][1]
    		px_b[n][i,j]= px_0[n][(i,j)][2]
    		px_gray[n][i,j]= (px_r[n][i,j]+px_g[n][i,j]+px_b[n][i,j])/3


xdata=np.outer(np.arange(0,w,1),np.ones(h))
ydata=np.outer(np.arange(0,h,1),np.ones(w))
ydata=ydata.T

zdata = []
for i in range(len(indizes)):
    zdata.append(np.array([px_r[i],px_g[i],px_b[i],px_gray[i]]))

my_cmap = plt.get_cmap('rainbow')
fig=[]
ax=[]

for j in range(len(indizes)):
    for i in range(4):
    	fig.append(plt.figure(figsize =(5,5)))
    	ax.append(plt.axes(projection ='3d'))
    	#ax.set_zlim3d( [ 0.78,0.8 ] )   
    	# ax_arr[i].set_xlabel('x[mm]')
    	# ax_arr[i].set_ylabel('y[mm]')
    	# ax_arr[i].set_zlabel('B/T')
    	ax[(j*4)+i].plot_surface(xdata,ydata,zdata[j][i],cmap=my_cmap)

for i in range(len(indizes)):    
    ax[(i*4)+0].set_title('ROT',fontweight='bold')
    ax[(i*4)+1].set_title('GRUEN',fontweight='bold')
    ax[(i*4)+2].set_title('BLAU',fontweight='bold')
    ax[(i*4)+3].set_title('GRAU',fontweight='bold')

plt.show()

""" Plot speichern """
farben = ["rot","gruen","blau","grau"]
versuchsname = "Grauwertematrizen"
for i in range(len(indizes)*4):
    fig[i].savefig("./4.Plots/%s_%s_%s.png"%(versuchsname,indizes[i//4],farben[i%4]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
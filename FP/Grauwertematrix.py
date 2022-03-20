# Marius Wenk, Fernando Grumpe

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits import mplot3d


w=640
h=480

def draw_line(a, b):
    m = [b[0] - a[0], b[1] - a[1]]

    def g(x):
        return np.multiply(x, m) + a

    line = []

    for x in range(w):
        p = g((x - a[0]) / m[0])
        if (p[1] < h and p[1] >= 0):
            p[1] = np.round(p[1])
            line.append(p)
    return line

indizes = [21,22,23,24,25,26,27,30,11,12,13,14]


im_arr=[]
px_arr=[]

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
    ax[(i*4)+0].set_title(f'Rot, Bild {indizes[i]:03d}',fontweight='bold')
    ax[(i*4)+1].set_title(f'Grün, Bild {indizes[i]:03d}',fontweight='bold')
    ax[(i*4)+2].set_title(f'Blau, Bild {indizes[i]:03d}',fontweight='bold')
    ax[(i*4)+3].set_title(f'Grau, Bild {indizes[i]:03d}',fontweight='bold')
    
""" Plot speichern """
farben = ["rot","gruen","blau","grau"]
versuchsname = "Grauwertematrizen"
for i in range(len(indizes)*4):
    fig[i].savefig("./4.Plots/%s_%s_%s.png"%(versuchsname,indizes[i//4],farben[i%4]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
    
""" Daten vorbereiten """
selected = [22]
a1 = [0]
b1 = [300]
a2 = [600]
b2 = [300]

x = []
y = []
line = []
for i in range(len(selected)):
    line.append(draw_line(np.array([a1[i], b1[i]]), np.array([a2[i], b2[i]])))

    xsingle = []
    ysingle = []

    for j in range(len(line[i])):
        xsingle.append(line[i][j][0])
        ysingle.append(line[i][j][1])
        
    x.append(xsingle)
    y.append(ysingle)

adata = []
bdata = []
for i in range(len(selected)):
    adata_single = []
    bdata_single = []
    n = indizes.index(selected[i])
    for j in range(len(line[i])):
        adata_single.append(zdata[n][3][int(x[i][j])][int(y[i][j])])
        bdata_single.append(np.sqrt((x[i][j]**2)+(y[i][j]**2)))
    adata.append(adata_single)
    bdata.append(bdata_single)
    
""" Plotten """
versuchsname = "TEM_Querschnitte"

fig = []
ax = []
for i in range(len(selected)):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].plot(bdata[i],adata[i],'o',label="Werte mit Fehler",markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel("Position entlang x'")
    ax[i].set_ylabel("Intensität I")
    ax[i].set_title(f"Bild {selected[i]:03d}")


# =============================================================================
# """ Regressionskurve """ 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(countFiles):
#     xmax.append(max(x_data[i]) + 5)   
#     xmin.append(min(x_data[i]) - 5)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B, C):
#     return A * np.asarray(np.exp(-2*((x-C)**2)/(B**2)))
# 
# fitRes = []
# perr = []
# x = sp.symbols('x')
# for i in range(countFiles):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[0.4, 5, 14.5]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     perr.append(np.sqrt(np.diag(pCov)))
#     A = fitRes[i][0][0].round(3)
#     B = abs(fitRes[i][0][1].round(2))
#     C = fitRes[i][0][2].round(2)
#     string = "$f_{%s}$(x) = %s * exp(-(2$(x-%s)^2$)/$%s^2$)"%(i+1,A,C,B)
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
# =============================================================================

""" Plot speichern """
for i in range(len(selected)):
    fig[i].savefig("./4.Plots/%s_0%s_plot.png"%(versuchsname,selected[i]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])

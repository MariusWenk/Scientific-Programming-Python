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
    	ax[(j*4)+i].set_xlabel("$x$ in Pixeln")
    	ax[(j*4)+i].set_ylabel("$y$ in Pixeln")
    	ax[(j*4)+i].set_zlabel("Intensität $I$")
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
versuchsname = "TEM_Querschnittspositionen"

selected = [22,21,23,24,26,30]
a1 = [0,0,0,80,0,0]
b1 = [210,300,260,0,125,70]
a2 = [600,600,600,510,600,600]
b2 = [190,300,260,400,190,320]
c1 = [0,0,0,500,0,465]
d1 = [410,320,280,0,350,0]
c2 = [500,500,500,0,500,200]
d2 = [430,320,280,300,310,500]

x1 = []
y1 = []
x2 = []
y2 = []
line1 = []
line2 = []
fig = []
ax = []
for i in range(len(selected)):
    line1.append(draw_line(np.array([a1[i], b1[i]]), np.array([a2[i], b2[i]])))
    line2.append(draw_line(np.array([c1[i], d1[i]]), np.array([c2[i], d2[i]])))
    
    if i in [0,1,2,4]:
        line2[i] = [(line2[i][j][1],line2[i][j][0]) for j in range(480)]

    xsingle = []
    ysingle = []

    for j in range(len(line1[i])):
        xsingle.append(line1[i][j][0])
        ysingle.append(line1[i][j][1])
        
    x1.append(xsingle)
    y1.append(ysingle)
    
    xsingle = []
    ysingle = []

    for j in range(len(line2[i])):
        xsingle.append(line2[i][j][0])
        ysingle.append(line2[i][j][1])
        
    x2.append(xsingle)
    y2.append(ysingle)
    
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].imshow(pic_0[indizes.index(selected[i])])
    ax[i].plot(x1[i],y1[i],color="Blue")
    ax[i].plot(x2[i],y2[i],color="Blue")
    ax[i].set_xlabel("$x$ in Pixeln")
    ax[i].set_ylabel("$y$ in Pixeln")
    
""" Plot speichern """
for i in range(len(selected)):
    fig[i].savefig("./4.Plots/%s_0%s_plot.png"%(versuchsname,selected[i]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])

adata1 = []
bdata1 = []
adata2 = []
bdata2 = []
for i in range(len(selected)):
    n = indizes.index(selected[i])
    adata_single = []
    bdata_single = []
    for j in range(len(line1[i])):
        adata_single.append(zdata[n][3][int(x1[i][j])][int(y1[i][j])])
        bdata_single.append(np.sqrt(((x1[i][j]-x1[i][0])**2)+((y1[i][j]-y1[i][0])**2)))
    adata1.append(adata_single)
    bdata1.append(bdata_single)
    
    adata_single = []
    bdata_single = []
    for j in range(len(line2[i])):
        adata_single.append(zdata[n][3][int(x2[i][j])][int(y2[i][j])])
        bdata_single.append(np.sqrt(((x2[i][j]-x2[i][0])**2)+((y2[i][j]-y2[i][0])**2)))
    adata2.append(adata_single)
    bdata2.append(bdata_single)
    
""" Plotten """
versuchsname = "TEM_Querschnitte"

fig = []
ax = []
for i in range(len(selected)):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].plot(bdata1[i],adata1[i],'o',label="Werte mit Fehler",markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel("Position entlang $x'$ in Pixeln")
    ax[i].set_ylabel("Intensität I")
    ax[i].set_title(f"Bild {selected[i]:03d}")
    
for i in range(len(selected),2*len(selected)):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].plot(bdata2[i-len(selected)],adata2[i-len(selected)],'o',label="Intensitätswerte",markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel("Position entlang $y'$ in Pixeln")
    ax[i].set_ylabel("Intensität I")
    ax[i].set_title(f"Bild {selected[i-len(selected)]:03d}")


""" Regressionskurve """ 
xmax = []
xmin = []
x_data_unlimited = []
for i in range(len(selected)):
    xmax.append(max(bdata1[i]) + 5)   
    xmin.append(min(bdata1[i]) - 5)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
   
for i in range(len(selected)):
    xmax.append(max(bdata2[i]) + 5)   
    xmin.append(min(bdata2[i]) - 5)
    x_data_unlimited.append(np.arange(xmin[i+len(selected)],xmax[i+len(selected)],0.01))

def H1(x):
    return 1

def H2(x):
    return 2*x

def H3(x):
    return (4*(x**2))-2

def fitCurve1(x, A, B, C):
    return A * (H1((np.sqrt(2)*np.asarray(x-C))/B)*np.asarray(np.exp(-((x-C)**2)/(B**2))))**2

def fitCurve2(x, A, B, C):
    return A * (H2((np.sqrt(2)*np.asarray(x-C))/B)*np.asarray(np.exp(-((x-C)**2)/(B**2))))**2

def fitCurve3(x, A, B, C):
    return A * (H3((np.sqrt(2)*np.asarray(x-C))/B)*np.asarray(np.exp(-((x-C)**2)/(B**2))))**2

fitRes = []
pFit = []

# i = 0
# fitRes.append(curve_fit(fitCurve2, adata1[i], bdata1[i], p0=[200, 140, 320]))

# i = 0
# fitRes.append(curve_fit(fitCurve1, adata2[i], bdata2[i], p0=[250, 240, 200]))

pFit.append([170, 145, 320])
pFit.append([250, 174, 330])
pFit.append([170, 153, 280])
pFit.append([170, 144, 340])
pFit.append([50, 120, 350])
pFit.append([50, 120, 380])

pFit.append([250, 260, 190])
pFit.append([250, 180, 300])
pFit.append([180, 162, 265])
pFit.append([170, 145, 200])
pFit.append([250, 240, 180])
pFit.append([54, 152, 305])


perr = []
string = []
for i in range(len(selected)*2):
    # pFit.append(fitRes[i][0])
    # pCov = fitRes[i][1]
    # perr.append(np.sqrt(np.diag(pCov)))
    A = pFit[i][0]
    B = pFit[i][1]
    C = pFit[i][2]
    string.append(f"Fitparameter: \n $A$ = {A} \n $B$ = {B} \n $C$ = {C}")
    # print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))

i = 0
ax[i].plot(x_data_unlimited[i], fitCurve2(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 1
ax[i].plot(x_data_unlimited[i], fitCurve1(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 2
ax[i].plot(x_data_unlimited[i], fitCurve2(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 3
ax[i].plot(x_data_unlimited[i], fitCurve2(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 4
ax[i].plot(x_data_unlimited[i], fitCurve3(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 5
ax[i].plot(x_data_unlimited[i], fitCurve3(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()

i = 0+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve1(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 1+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve1(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 2+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve2(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 3+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve2(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 4+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve1(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()
i = 5+len(selected)
ax[i].plot(x_data_unlimited[i], fitCurve3(x_data_unlimited[i], *pFit[i]), label=string[i],linewidth=2)
ax[i].legend()

""" Plot speichern """
for i in range(len(selected)):
    fig[i].savefig("./4.Plots/%s_x'_0%s_plot.png"%(versuchsname,selected[i]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Plot speichern """
for i in range(len(selected)):
    fig[i+len(selected)].savefig("./4.Plots/%s_y'_0%s_plot.png"%(versuchsname,selected[i]), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i+len(selected)])

# Marius Wenk, Pau Gran ́e i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 5
versuchsname = "P4"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    fin = open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap")
    fout = open("%s_%s_data_mod.csv"%(versuchsname,i), "w", encoding="charmap")
    for line in fin:
        fout.write(line.replace(',', '.').replace(";;;;;;;;","").replace(";;;;;;;",""))
    fin.close()
    fout.close()
    file.append(open("%s_%s_data_mod.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """


""" Daten vorbereiten """
s = []
x_data = []
y_data = []
#xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    #xerr.append([0.02 for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([0.001 for j in range(s[i])])

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
ax[0].set_title("$T$ = 30,3°C")
ax[1].set_title("$T$ = 21,0°C")
ax[2].set_title("$T$ = 3,6°C")
ax[3].set_title("$T$ = -10,3°C")
ax[4].set_title("$T$ = -21,9°C")

for i in range(countFiles):
    ax[i].set_xlabel("$t$ (s)")
    ax[i].set_ylabel("$V$ (mV)")

""" Regressionskurve """ 
x_data_unlimited = []
x_data_select = []
y_data_select = []
s1  = [40,90,50,70,110]
s2  = [400,390,450,400,400]
for i in range(countFiles):
    mask = np.ones(x_data[i].size, dtype=bool)
    mask[s1[i]:s2[i]] = False
    x_data_unlimited.append(np.arange(0,x_data[i][-1],2))
    x_data_select.append(x_data[i][mask])
    y_data_select.append(y_data[i][mask])

def fitCurve(x, A, B, C):
    return (A * (np.asarray(x)**2)) + (B * np.asarray(x)) + C


perr = []
fitRes = []
pFit = []
pCov = []
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data_select[i], y_data_select[i], p0=[0.5,-1,0]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(countFiles):
    A = fitRes[i][0][0].round(10)
    B = fitRes[i][0][1].round(5)
    C = fitRes[i][0][2].round(4)
    fitCurveSym = A * (x**2) + B * x + C 
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    string = "Linia base: %s"%fitCurveSym
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), label=string, linewidth=2, color="blue")
    ax[i].legend()
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Weitere Plots"""
y_data_new = []
Y = []
for i in range(countFiles):
    y_data_new.append(y_data[i]-fitCurve(x_data[i], *pFit[i]))
    fig.append(plt.figure())
    ax.append(fig[i+countFiles].add_axes([0.15,0.15,0.75,0.75]))
    Y.append(np.amax(y_data_new[i]).round(4))
    string = "$Y$ = %s mV"%Y[i]
    ax[i+countFiles].plot(x_data[i],y_data_new[i],'o',label=string,markersize=2,color="Black")
    ax[i+countFiles].legend()
    ax[i+countFiles].grid(True)
Y = np.flip(Y)
    
ax[5].set_title("$T$ = 30,3°C")
ax[6].set_title("$T$ = 21,0°C")
ax[7].set_title("$T$ = 3,6°C")
ax[8].set_title("$T$ = -10,3°C")
ax[9].set_title("$T$ = -21,9°C")

for i in range(countFiles, 2*countFiles):
    ax[i].set_xlabel("$t$ (s)")
    ax[i].set_ylabel("$V$ (mV)")
    
T = [-21.9,-10.3,3.6,21.0,30.3]
fig.append(plt.figure())
ax.append(fig[10].add_axes([0.15,0.15,0.75,0.75]))
ax[10].plot(T,Y,'x',markersize=4,color="Black")
ax[10].grid(True)
ax[10].set_xlabel("$T$ (°C)")
ax[10].set_ylabel("$Y(T)$ (mV)")

I = [-2.13,-2.07,-2.15,-1.92,-2.14]
S = []
for i in range(5):
    S.append(Y[i]/(I[i]*2))
fig.append(plt.figure())
ax.append(fig[11].add_axes([0.15,0.15,0.75,0.75]))
ax[11].plot(T,S,'x',markersize=4,color="Black")
ax[11].grid(True)
ax[11].set_xlabel("$T$ (°C)")
ax[11].set_ylabel("$S(T)$ (mV/W)")

def fitLine(x, A, B):
    return (A * np.asarray(x)) + B

fitRes.append(curve_fit(fitLine, T, S, p0=[0.5,-1]))
pFit.append(fitRes[5][0])
pCov.append(fitRes[5][1])

x_data_unlimited.append(np.arange(-25,35,1))
x = sp.symbols('x')
A = fitRes[5][0][0].round(4)
B = fitRes[5][0][1].round(4)
fitCurveSym = A * x + B 
print("Regressionskurve %s: %s"%(5+1,fitCurveSym))
string = "Fit: %s"%fitCurveSym
ax[11].plot(x_data_unlimited[5], fitLine(x_data_unlimited[5], *pFit[5]), label=string, linewidth=2, color="blue")
ax[11].legend()
    
""" Daten in Tabelle ausgeben """
b = []
b.append(np.around(T,3))
b.append(np.around(I,3))
b.append(np.around(Y,3))
b.append(np.around(S,3))
labels = ["$T$ (°C)","$I(T)$ (A)","$Y(T)$ (mV)","$S(T)$ (mV/W)"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[12].add_axes([0,0,1,1]))
ax[12].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[12].axis("off")
ax[12]


""" Plot speichern """
for i in range((2*countFiles)+3):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



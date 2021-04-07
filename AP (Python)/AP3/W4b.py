# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 5
versuchsname = "W4b"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))
beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))
    
""" Konstanten """
boltz = 5.6704e-8
# A = 15

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append([0.02 for i in range(s)])
    y_data.append(beamData[i][:,2])
    yerr.append([0.02 for i in range(s)])

""" Plotten """
fig = []
ax = []
for i in range(countFiles+2):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
ax[0].set_xlabel("$T$ in °C")
ax[0].set_ylabel("$U$ in mV")
for i in range(1,5):
    ax[i].set_xlabel("$T$ in K")
    ax[i].set_ylabel("$P$ in W")

""" Regressionskurve """ 
x_data_unlimited = np.arange(-0.1,5,0.001)

def fitCurve(x, A, B):
    return A * (np.asarray(x)**4)

perr = []
fitRes = []
for i in range(2,4):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1,4]))
    pFit = fitRes[i-2][0]
    pCov = fitRes[i-2][1]
    ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i-2])
ax[2].set(xlim=(0.0027,0.0035))
ax[2].set(ylim=(-2,3))
ax[3].set(xlim=(-0.05,5))
# ax[3].set(ylim=(-2,3))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(2,4):
    A = fitRes[i-2][0][0].round(4)
    B = fitRes[i-2][0][1].round(4)
    fitCurve = A * x + B
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Daten in Tabelle ausgeben """
for i in range(3,4):
    b = []
    b.append(np.around(x_data[i],3))
    b.append(np.around(y_data[i],3))
    b.append(np.around(yerr[i],3))
    labels = ["$P$ in W","$T$ in °C","$\Delta T$ in °C"]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i+1].add_axes([0,0,1,1]))
    ax[i+1].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i+1].axis("off")
    ax[i+1]

""" Plot speichern """
for i in range(len(y_data)+1):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert

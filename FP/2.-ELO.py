# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 8
versuchsname = "V2"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("2.Daten/%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))
beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))
    
""" Konstanten """

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
err = [0.01,0.02,0.02]
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append([err[i//3] for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([2 for j in range(s[i])])

x_data[3] = x_data[3][::-1]
y_data[3] = y_data[3][::-1]

""" Plotten """
versuchsname = "ELO"

fig = []
ax = []
stri = ["$I_H$","$U_W$","$U_W$"]
val = ["2,20A","2,51A","2,70A","0,00V","0,00V","3,01V","6,00V","1,50V"]
xlab = ["$U_W$ in V","$I_H$ in A","$I_H$ in A"]
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel(f"{xlab[i//3]}")
    ax[i].set_ylabel("$I_S$ in $\mu$A")
    ax[i].set_title(f"{stri[i//3]} = {val[i]}, {i//4+1}. Kathode")

""" Regressionskurve """ 
xmax = [2.8,2.8,2.8,2.8]
xmin = [2.0,2.45,2.45,2.45]
indize = [4,10,8,7]
fitplots = [3,4,5,7]
x_data_unlimited = []
for i in range(4):
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B):
    return A * np.asarray(x-B)

fitRes = []
perr = []
for i in range(4):
    fitRes.append(curve_fit(fitCurve, x_data[fitplots[i]][indize[i]:], y_data[fitplots[i]][indize[i]:], p0=[-1, 1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[fitplots[i]].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[fitplots[i]].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[fitplots[i]].set(xlim=(xmin[i],xmax[i]))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(4):
    A = fitRes[i][0][0].round(4)
    B = fitRes[i][0][1].round(4)
    fitCurve = A * x - B
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Regressionskurve """ 
x_data_unlimited = []
for i in range(4):
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B):
    return A * np.asarray(x-B)

fitRes = []
perr = []
for i in range(4):
    fitRes.append(curve_fit(fitCurve, x_data[fitplots[i]][indize[i]:], y_data[fitplots[i]][indize[i]:], p0=[-1, 1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[fitplots[i]].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[fitplots[i]].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[fitplots[i]].set(xlim=(xmin[i],xmax[i]))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(4):
    A = fitRes[i][0][0].round(4)
    B = fitRes[i][0][1].round(4)
    fitCurve = A * x - B
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./2.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
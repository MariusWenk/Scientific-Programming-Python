# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
# from sympy import *

""" Daten auslesen """
countFiles = 2
versuchsname = "A1b"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))

""" Konstanten """
lambdas = [405,436,546,578] 
transms = ["0,05","0,1","0,25","0,5"]

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append([0.025 for j in range(s[i])])
    for j in range(1,5):
        y_data.append(beamData[i][:,j])
        yerr.append(beamData[i][:,j+4])

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    for j in range(4):
        if (i == 0):
            ax[i].errorbar(x_data[i],y_data[j],yerr[j],xerr[i],label="$\lambda$ = (%s $\pm$ 0,5)nm"%lambdas[j],fmt='o',markersize=2)
        else:
            ax[i].errorbar(x_data[i],y_data[j+4],yerr[j+4],xerr[i],label="Transm. = (%s $\pm$ 0,01)"%transms[j],fmt='o',markersize=2)
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel("Gegenspannung $U_g$ in V")
    ax[i].set_ylabel("Anodenspannung $U_a$ in mV (~$I_{Ph}$)")
    #ax[i].axis([2,8,4.5,7.6])

ax[1].set_title("$\lambda$ = (405 $\pm$ 0,5)nm")

""" Regressionskurve """
# def fitCurve(x, A, B):
#     return A * np.asarray(x) + B

# fitRes = []
# for i in range(4):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1,4]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     ax[i].plot(x_data[i], fitCurve(x_data[i], *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
#     perr = np.sqrt(np.diag(pCov))
#     print(pCov)
#     print(perr)

""" Regressionsfunktion """
# x = Symbol('x')
# A = pFit[0].round(4)
# B = pFit[1].round(4)
# fitCurve1 = A * x + B
# print(fitCurve1)

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./Plots/%s_%s_zusammen_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert


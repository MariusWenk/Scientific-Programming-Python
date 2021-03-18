# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
# from scipy.optimize import curve_fit
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

""" Regressionskurve """
# def fitCurve(x, A, B):
#     return A * np.asarray(x) + B

# pFit, pCov = curve_fit(fitCurve, x_dataPos1, y_dataPos1, p0=[-1,4])
# plt.plot(x_dataPos1, fitCurve(x_dataPos1, *pFit), label="Fitkurve durch kleinste Quadrate", color="Orange",linewidth=3)
# perr = np.sqrt(np.diag(pCov))
# print(perr)

""" Regressionsfunktion """
# x = Symbol('x')
# A = pFit[0].round(4)
# B = pFit[1].round(4)
# fitCurve1 = A * x + B
# print(fitCurve1)

""" Plotten """
fig = []
ax = []
for i in range(len(y_data)):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i//4],y_data[i],yerr[i],xerr[i//4],label="(proportional zu) Photostrom-Gegenspannungskennlinie",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel("Gegenspannung $U_g$ in V")
    ax[i].set_ylabel("Anodenspannung $U_a$ in mV (~$I_{Ph}$)")
    #ax[i].axis([2,8,4.5,7.6])
   
lambdas = [405,436,546,578]  
for i in range(4):
    ax[i].set_title("$\lambda$ = (%s $\pm$ 0,5)nm"%lambdas[i])
    
transms = ["0,05","0,1","0,25","0,5"]
for i in range(4,8):
    ax[i].set_title("Transm. = (%s $\pm$ 0,01) bei $\lambda$ = (405 $\pm$ 0,5)nm"%transms[i-4])

for i in range(len(y_data)):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
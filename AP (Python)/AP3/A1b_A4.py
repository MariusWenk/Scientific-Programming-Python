# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
# from scipy.optimize import curve_fit
# from sympy import *

""" Daten auslesen """
countFiles = 1
versuchsname = "A1b"
ersterDatenIndex = 3
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))

""" Konstanten """
h_x = np.arange(2,8,0.01)
h = []
for i in range(h_x.size):
    h.append(6.626)

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append(beamData[i][:,1])
    y_data.append(beamData[i][:,2])
    yerr.append(beamData[i][:,3])

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
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].plot(h_x,h, color="Blue", label="$h_{Lit}$")
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],fmt='o',markersize=2,label="",color="Black")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel("Frequenz des Lichts in 10^14 Hz")
    ax[i].set_ylabel("$h$ in 10^-34 J")
    ax[i].axis([2,8,4.5,7.6])
    
#ax[0].set_title("1-0-0 Ebene")

for i in range(countFiles):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert

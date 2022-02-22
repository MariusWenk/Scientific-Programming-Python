# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 1
versuchsname = "A1b"
ersterDatenIndex = 4
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
    xerr.append(beamData[i][:,2])
    y_data.append(beamData[i][:,1])
    yerr.append(beamData[i][:,3])

""" Plotten """
fig = []
ax = []
for i in range(len(y_data)):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i//4],y_data[i],yerr[i],xerr[i//4],label="",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel("Frequenz $ny$ in 10^14 Hz")
    ax[i].set_ylabel("$eU_0$ in eV")
    # ax[i].axis([0,1,2,3])
    ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
""" Regressionskurve """ 
x_data_unlimited = np.arange(-1,8,0.01)

def fitCurve(x, A, B):
    return A * np.asarray(x) + B

perr = []
fitRes = []
for i in range(1):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1,4]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(1):
    A = fitRes[i][0][0].round(4)
    B = fitRes[i][0][1].round(4)
    fitCurve = A * x + B
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    lambdified_fitCurve = sp.lambdify(x,fitCurve)
    #Nulstellen:
    print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))

""" Plot speichern """
for i in range(len(y_data)):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,9), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
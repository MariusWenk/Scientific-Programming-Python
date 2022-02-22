# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

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
    # ax[i].axis([0,1,2,3])
    ax[i].set(xlim=(-0.2,2.2))
    # ax[i].set(ylim=(-0.2,2.2))
   
lambdas = [405,436,546,578]  
for i in range(4):
    ax[i].set_title("$\lambda$ = (%s $\pm$ 0,5)nm"%lambdas[i])
    
transms = ["0,05","0,1","0,25","0,5"]
for i in range(4,8):
    ax[i].set_title("Transm. = (%s $\pm$ 0,01) bei $\lambda$ = (405 $\pm$ 0,5)nm"%transms[i-4])
    
""" Regressionskurve """
lim = [6,5,3,3]
x_data_limited = []
for i in range(4):
    x_data_limited.append(x_data[0][:lim[i]])
y_data_limited = []
for i in range(4):
    y_data_limited.append(y_data[i][:lim[i]])
    
x_data_unlimited = np.arange(-1,3,0.01)

def fitCurve(x, A, B):
    return A * np.asarray(x) + B

perr = []
fitRes = []
for i in range(4):
    fitRes.append(curve_fit(fitCurve, x_data_limited[i], y_data_limited[i], p0=[-1,4]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(4):
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
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
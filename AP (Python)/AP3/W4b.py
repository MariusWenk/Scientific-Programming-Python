# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 6
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
Fl = 7.85e-5
dFl = 0.79e-5
E = 0.16
dT = 0.1
dU = 0.02
B = 19.071
dB = 0.056

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
s.append(beamData[0][:,0].size)
x_data.append(beamData[0][:,2])
xerr.append([0.02 for j in range(s[0])])
y_data.append(beamData[0][:,1]-beamData[0][:,0])
yerr.append([0.4 for j in range(s[0])])
for i in range(1,6):
    s.append(beamData[i][:,0].size)
    x_data.append((B*beamData[i][:,0])+beamData[i][:,2]+273.15)
    xerr.append((dB*beamData[i][:,0])+B*dU+dT)
    y_data.append(beamData[i][:,1]/E)
    yerr.append([0.125 for j in range(s[i])])

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
ax[0].set_xlabel("$U_{T}$ in mV")
ax[0].set_ylabel("$dT$ in K")
for i in range(1,6):
    ax[i].set_xlabel("$T_{P}$ in K")
    ax[i].set_ylabel("$P$ in mW")

""" Regressionskurve """ 
x_data_unlimited_line = np.arange(-0.1,5,0.01)
x_data_unlimited = np.arange(290,375,0.1)

def fitLine(x, A):
    return A * np.asarray(x)

def fitCurve(x, A):
    return A * (np.asarray(x)**4)

perr = []
fitRes = []
fitRes.append(curve_fit(fitLine, x_data[0], y_data[0], p0=[-1]))
pFit = fitRes[0][0]
pCov = fitRes[0][1]
ax[0].plot(x_data_unlimited_line, fitLine(x_data_unlimited_line, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
ax[0].legend()
perr.append(np.sqrt(np.diag(pCov)))
print("Fitfehler",0+1,perr[0])
ax[0].set(xlim=(0,4.5))
for i in range(1,6):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
A = fitRes[0][0][0].round(4)
fitCurve = A * x 
print("Regressionskurve %s: %s"%(0+1,fitCurve))
for i in range(1,6):
    A = fitRes[i][0][0].round(14)
    fitCurve = A * x**4
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    print("Emissionsgrad %s: %s"%(i+1,A*1e-3/(Fl*boltz)))
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Daten in Tabelle ausgeben """
for i in range(1,6):
    b = []
    b.append(np.around(x_data[i],3))
    b.append(np.around(xerr[i],3))
    b.append(np.around(y_data[i],3))
    labels = ["$T_{P}$ in K","$\Delta T_{P}$ in K","$P$ in mW"]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i+5].add_axes([0,0,1,1]))
    ax[i+5].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i+5].axis("off")
    ax[i+5]

""" Plot speichern """
for i in range(11):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert

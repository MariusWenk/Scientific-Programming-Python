# Marius Wenk, Pau GranÃ© i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 1
versuchsname = "P2"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
del_t = 2
del_m = 0.01
del_V = 0.2
del_I = 0.01
m_pot = 203.72


""" Daten vorbereiten """


s = []
x_data = []
diff_m = []
y_data = []
xerr = []
yerr = []
s = beamData[i][:,0].size
for i in range(s):
    if (i == 0):
        diff_m.append(beamData[0][i,2]-m_pot)
    else:
        diff_m.append(beamData[0][i,2]-beamData[0][i-1,2])
    x_data.append(diff_m[i]/(beamData[0][i,3]*60))
    xerr.append(((beamData[0][i,2]*del_t)/((beamData[0][i,3]*60)**2)) + ((2*del_m)/(beamData[0][i,3]*60)))
    y_data.append(beamData[0][i,0] * beamData[0][i,1])
    yerr.append((beamData[0][i,0] * del_I) + (del_V * beamData[0][i,1]))

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data,y_data,yerr,xerr,fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))

ax[0].set_xlabel("$\Delta m/\Delta t$ (g/s)")
ax[0].set_ylabel("$P$ (W)")

""" Regressionskurve """
def fitCurve(x, A, B):
    return (A * np.asarray(x)) + B

perr = []
fitRes = []
pFit = []
pCov = []
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data, y_data, p0=[0.5,1]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
x_data_unlimited = []
for i in range(1):
    x_data_unlimited.append(np.arange(0,0.1,0.001))
    A = fitRes[i][0][0].round(5)
    B = fitRes[i][0][1].round(5)
    fitCurveSym = A * x + B
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    string = "Función de calibración: %s"%fitCurveSym
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), label=string, linewidth=2, color="blue")
    ax[i].legend()
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
    
""" Daten in Tabelle ausgeben """
b = []
b.append(np.around(beamData[0][:,0],3))
b.append(np.around(beamData[0][:,1],3))
b.append(np.around(beamData[0][:,2],3))
b.append(np.around(beamData[0][:,3],3))
b.append(np.around(diff_m,3))
b.append(np.around(x_data,3))
b.append(np.around(xerr,3))
b.append(np.around(y_data,3))
b.append(np.around(yerr,3))
labels = ["$V$ (V)","$I$ (A)","$m_{tot}$ (g)","$\Delta t$ (min)", "$\Delta m$ (g)", "$\Delta m/\Delta t$ (g/s)", "$\delta (\Delta m/\Delta t)$ (g/s)", "$P$ (W)", "$\delta P$ (W)"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[1].add_axes([0,0,1,1]))
ax[1].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[1].axis("off")

""" Plot speichern """
for i in range(2):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



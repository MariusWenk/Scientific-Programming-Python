# Marius Wenk, Pau Grane i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 1
versuchsname = "P6"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
err_T = 1
err_p = 0.05


""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append([((beamData[0][j,0])) for j in range(s[i])])
    xerr.append([err_T for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([err_p for j in range(s[i])])
    
for i in range(2):
    s.append(beamData[0][:,0].size)
    x_data.append([(1/(beamData[0][j,0]+273.15)) for j in range(s[i])])
    xerr.append([err_T/((beamData[0][j,0]+273.15)**2) for j in range(s[i])])
    y_data.append(np.log(beamData[0][:,1]))
    yerr.append([err_p/beamData[0][j,1] for j in range(s[i])])

""" Plotten """
fig = []
ax = []
for i in range(3):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],fmt='o',markersize=2,color="Black")
    #ax[i].plot(x_data[i],y_data[i],linewidth=1,color="Blue")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
string = "Punto critico: $T$ = %s°C; $p$ = %sMPa"%(x_data[0][6],y_data[0][6])
ax[0].scatter(x_data[0][6],y_data[0][6],label=string,marker="X",s=200,color="Green")
ax[0].legend()

ax[0].set_xlabel("$T$ (°C)")
ax[0].set_ylabel("$p$ (MPa)")
ax[1].set_xlabel("$1/T$ (1/K)")
ax[1].set_ylabel("$ln(p/$MPa$)$")
ax[2].set_xlabel("$1/T$ (1/K)")
ax[2].set_ylabel("$ln(p/$MPa$)$")

#ax[1].set_title("sin curva de ajuste")
#ax[2].set_title("con curva de ajuste")

""" Regressionskurve """
def fitCurve(x, A, B):
    return (A * np.asarray(x)) + B

perr = []
fitRes = []
pFit = []
pCov = []
x_data_limited = []
y_data_limited = []
x_data_limited.append([(1/(beamData[0][j,0]+273.15)) for j in range(0,s[i])])
y_data_limited.append(np.log(beamData[0][0:,1]))
for i in range(1):
    fitRes.append(curve_fit(fitCurve,x_data_limited[i], y_data_limited[i], p0=[0.5,1]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
x_data_unlimited = []
x_data_unlimited.append(np.arange(x_data_limited[0][-1],x_data_limited[0][0],0.000001))
#x_data_unlimited.append(np.arange(0.02,0.045,0.001))
for i in range(1):
    A = fitRes[i][0][0].round(5)
    B = fitRes[i][0][1].round(5)
    fitCurveSym = A * x + B
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    ax[i+2].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), linewidth=2, color="blue")
    ax[i+2].legend()
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Daten in Tabelle ausgeben """
# =============================================================================
# b = []
# b.append(np.around(beamData[0][:,0],3))
# b.append(np.around(beamData[0][:,1],3))
# b.append(np.around(beamData[0][:,2],3))
# b.append(np.around(beamData[0][:,3],3))
# b.append(np.around(diff_m,3))
# b.append(np.around(x_data,3))
# b.append(np.around(xerr,3))
# b.append(np.around(y_data,3))
# b.append(np.around(yerr,3))
# labels = ["$V$ (V)","$I$ (A)","$m_{tot}$ (g)","$\Delta t$ (min)", "$\Delta m$ (g)", "$\Delta m/\Delta t$ (g/s)", "$\delta (\Delta m/\Delta t)$ (g/s)", "$P$ (W)", "$\delta P$ (W)"]
# b = np.array(b).T
# fig.append(plt.figure())
# ax.append(fig[1].add_axes([0,0,1,1]))
# ax[1].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
# ax[1].axis("off")
# =============================================================================

""" Plot speichern """
for i in range(3):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



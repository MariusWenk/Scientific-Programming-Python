# Marius Wenk, Pau Grane i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 6
versuchsname = "P5"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
err_m = 0.01


""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i].size)
    x_data.append([(2*j) for j in range(s[i])])
    #xerr.append([0.02 for j in range(s[i])])
    y_data.append(beamData[i])
    yerr.append([0.01 for j in range(s[i])])

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],fmt='o',markersize=2,color="Black")
    #ax[i].plot(x_data[i],y_data[i],linewidth=1,color="Blue")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))

for i in range(countFiles):
    ax[i].set_xlabel("$Tiempo$ (s)")
    ax[i].set_ylabel("$m$ (mg)")

for i in range(3):
    ax[i].set_title("Grande, Mesura %s"%(i+1))
    
for i in range(3, countFiles):
    ax[i].set_title("Pequeño, Mesura %s"%(i-2))

""" Regressionskurve """
# =============================================================================
# def fitCurve(x, A):
#     return A
# 
# s1 = [8,8,8]
# s2 = [19,12,21]
# s3 = [34,17,33]
# 
# perr = []
# fitRes = []
# pFit = []
# pCov = []
# for i in range(countFiles):
#     fitRes.append(curve_fit(fitCurve, x_data[i][:s1[i]], y_data[i][:s1[i]], p0=[0.5]))
#     fitRes.append(curve_fit(fitCurve, x_data[i][s2[i]:s3[i]], y_data[i][s2[i]:s3[i]], p0=[0.5]))
#     pFit.append(fitRes[(2*i)][0])
#     pCov.append(fitRes[(2*i)][1])
#     pFit.append(fitRes[(2*i)+1][0])
#     pCov.append(fitRes[(2*i)+1][1])
#     perr.append(np.sqrt(np.diag(pCov[(2*i)])))
#     print("Fitfehler unten",i+1,perr[(2*i)])
#     perr.append(np.sqrt(np.diag(pCov[(2*i)+1])))
#     print("Fitfehler oben",i+1,perr[(2*i)+1])
#     #ax[i].set(xlim=(295,370))
# =============================================================================

""" Regressionsfunktion """
# =============================================================================
# x = sp.symbols('x')
# x_data_unlimited = []
# for i in range(countFiles):
#     x_data_unlimited.append(np.arange(-100,x_data[i][-1],0.1))
#     A1 = fitRes[(2*i)][0][0].round(3)
#     A2 = fitRes[(2*i)+1][0][0].round(3)
#     print("Regressionskurve %s: %s"%(i+1,A1))
#     print("Regressionskurve %s: %s"%(i+1,A2))
#     ax[i].plot(x_data_unlimited[i], [A1 for j in range(x_data_unlimited[i].size)], linewidth=2, color="blue")
#     ax[i].plot(x_data_unlimited[i], [A2 for j in range(x_data_unlimited[i].size)], linewidth=2, color="blue")
#     # ax[i].legend()
#     # lambdified_fitCurve = sp.lambdify(x,fitCurve)
#     # #Nulstellen:
#     # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
#     # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
#     # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
# =============================================================================
    
    
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
for i in range(countFiles):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



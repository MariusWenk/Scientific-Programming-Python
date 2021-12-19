# Marius Wenk, Pau Grane i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 3
versuchsname = "P8"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
err_h = [0.5, 0.5, 0.5]

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append([err_h[i] for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([err_h[i] for j in range(s[i])])
        
""" Daten in CSV schreiben """
# =============================================================================
# for i in range(2):
#     f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i), "w")
#     writer = csv.writer(f)
#     writer.writerow(x_data[i])
#     for j in range(mesuras):
#         writer.writerow(y_data[j+(i*mesuras)])
#     f.close
#     
# for i in range(2,4):
#     f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i), "w")
#     writer = csv.writer(f)
#     writer.writerow(x_data[i])
#     writer.writerow(y_data[i+(2*mesuras)-2])
#     f.close
#     
# for i in range(2):
#     f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i+4), "w")
#     writer = csv.writer(f)
#     writer.writerow(x_data[i])
#     for j in range(mesuras):
#         writer.writerow(yerr[j+(i*mesuras)])
#     f.close
#     
# for i in range(2,4):
#     f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i+4), "w")
#     writer = csv.writer(f)
#     writer.writerow(x_data[i])
#     writer.writerow(yerr[i+(2*mesuras)-2])
#     f.close
# =============================================================================

""" Plotten """
fig = []
ax = []
for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
  
string = ["Argón", "Aire", "$CO_2$"]
color = ["Blue", "Orange", "Green"]

fig.append(plt.figure())
ax.append(fig[countFiles].add_axes([0.15,0.15,0.75,0.75]))
for i in range(countFiles):
    ax[countFiles].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],fmt='o',label=string[i],markersize=2,color=color[i])
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
ax[countFiles].grid(True)
ax[countFiles].legend()

for i in range(countFiles+1):
    ax[i].set_xlabel("$h_1$ (cm)")
    ax[i].set_ylabel("$h_2$ (cm)")

for i in range(countFiles):
    ax[i].set_title(string[i])
    
ax[countFiles].set_title("Comparación")

""" Regressionskurve """
def fitCurve(x, A):
    return (A * np.asarray(x))

perr = []
fitRes = []
pFit = []
pCov = []
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve,x_data[i], y_data[i], p0=[2]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
x_data_unlimited = []
x_data_unlimited.append(np.arange(0,45,0.1))
for i in range(countFiles):
    A = fitRes[i][0][0].round(5)
    fitCurveSym = A * x
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    ax[countFiles].plot(x_data_unlimited[0], fitCurve(x_data_unlimited[0], *pFit[i]), linewidth=2, color=color[i])
    ax[i].plot(x_data_unlimited[0], fitCurve(x_data_unlimited[0], *pFit[i]), linewidth=2, color="Blue")
    ax[countFiles].legend()
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
for i in range(countFiles+1):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



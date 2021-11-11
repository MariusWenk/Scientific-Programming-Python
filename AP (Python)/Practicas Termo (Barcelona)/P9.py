# Marius Wenk, Pau Grane i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import csv
import math

""" Daten auslesen """
countFiles = 4
versuchsname = "P9"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
T_amb = 22.7
err_eps = 0.01
off_eps = 0.011
a = 0.0366
err_a = 0.0014
b = 3.3e-5
err_b = 1.6e-5

""" Daten vorbereiten """
s = []
x_data = []
eps_data = []
y_data = []
xerr = []
yerr = []
tiempos = []
mesuras = 0
for i in range(2):
    s.append(beamData[i][0,:].size)
    x_data.append([beamData[i][0,j] for j in range(1,s[i])])
    mesuras = beamData[i][:,0].size - 1
    tiempos = []
    for k in range(mesuras):
        tiempos.append(beamData[i][k+1,0])
        eps_data.append([beamData[i][k+1,j] - off_eps for j in range(1,s[i])])
        
for i in range(2,4):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    eps_data.append([beamData[i][j,1] - off_eps for j in range(s[i])])
    
for i in range((2*mesuras)+2):
    if i < (mesuras*2):
        c = 4
    else:
        c = s[i - (2*mesuras) + 2]
    y_data.append([-(a/(2*b)) + math.sqrt(((a/(2*b))**2) + (eps_data[i][j]/b)) for j in range(c)])
    yerr.append([0.1 for j in range(c)])
    #yerr.append([((err_a/(2*b)) + ((2*a*err_a)/((2*b)**2))/(np.sqrt(((a/(2*b))**2) + (eps_data[i][j]/b)))) + (((a*err_b)/(2*(b**2))) + ((2*a*a*err_b)/(((2*b)**2)*b))/(np.sqrt(((a/(2*b))**2) + (eps_data[i][j]/b)))) + ((err_eps)/(np.sqrt(((a/(2*b))**2) + (eps_data[i][j]/b)))) for j in range(c)])
    
""" Daten in CSV schreiben """
for i in range(2):
    f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i), "w")
    writer = csv.writer(f)
    writer.writerow(x_data[i])
    for j in range(mesuras):
        writer.writerow(y_data[j+(i*mesuras)])
    f.close
    
for i in range(2,4):
    f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i), "w")
    writer = csv.writer(f)
    writer.writerow(x_data[i])
    writer.writerow(y_data[i+(2*mesuras)-2])
    f.close
    
for i in range(2):
    f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i+4), "w")
    writer = csv.writer(f)
    writer.writerow(x_data[i])
    for j in range(mesuras):
        writer.writerow(yerr[j+(i*mesuras)])
    f.close
    
for i in range(2,4):
    f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i+4), "w")
    writer = csv.writer(f)
    writer.writerow(x_data[i])
    writer.writerow(yerr[i+(2*mesuras)-2])
    f.close

""" Plotten """
fig = []
ax = []
for i in range(2):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    for j in range(mesuras):
        string = "$t$ = %s min"%tiempos[j]
        ax[i].plot(x_data[i],y_data[(i*mesuras)+j],'X',label=string,markersize=10)
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
for i in range(2,4):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[(2*mesuras)+i-2],yerr[(2*mesuras)+i-2],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
ln_theta = []
err_ln_theta = []
for i in range(2):
    ln_theta.append(np.log(y_data[(2*mesuras)+i]))
    err_ln_theta.append([yerr[(2*mesuras)+i][j]/y_data[(2*mesuras)+i][j] for j in range(s[i+2])])
    fig.append(plt.figure())
    ax.append(fig[i+4].add_axes([0.15,0.15,0.75,0.75]))
    ax[i+4].errorbar(x_data[i+2],ln_theta[i],err_ln_theta[i],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i+4].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))

for i in range(4):
    ax[i].set_xlabel("$x$ (posiciones)")
    ax[i].set_ylabel("$T$ (°C)")

for i in range(4,6):
    ax[i].set_xlabel("$x$ (posiciones)")
    ax[i].set_ylabel("$ln(\Theta)$ (°C)")

ax[0].set_title("Cu")
ax[1].set_title("Fe")
ax[2].set_title("Cu equilibrio ($t$ = 55 min)")
ax[3].set_title("Fe equilibrio ($t$ = 55 min)")
ax[4].set_title("Cu equilibrio ($t$ = 55 min)")
ax[5].set_title("Fe equilibrio ($t$ = 55 min)")

""" Regressionskurve """
def fitCurve(x, A, B):
    return (A * np.asarray(x)) + B

perr = []
fitRes = []
pFit = []
pCov = []
x_data_limited = []
ln_theta_limited = []
x_data_limited.append(x_data[2][0:15])
x_data_limited.append(x_data[3][0:10])
ln_theta_limited.append(ln_theta[0][0:15])
ln_theta_limited.append(ln_theta[1][0:10])
for i in range(2):
    fitRes.append(curve_fit(fitCurve,x_data_limited[i], ln_theta_limited[i], p0=[0.5,1]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
x_data_unlimited = []
x_data_unlimited.append(np.arange(0,15,0.1))
x_data_unlimited.append(np.arange(0,10,0.1))
for i in range(2):
    A = fitRes[i][0][0].round(5)
    B = fitRes[i][0][1].round(5)
    fitCurveSym = A * x + B
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    ax[i+4].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), linewidth=2, color="blue")
    ax[i+4].legend()
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
for i in range(countFiles+2):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



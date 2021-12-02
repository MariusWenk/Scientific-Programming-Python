# Marius Wenk, Pau GranÃ© i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import csv

""" Daten auslesen """
countFiles = 2
versuchsname = "P7"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    fin = open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap")
    fout = open("%s_%s_data_mod.csv"%(versuchsname,i), "w", encoding="charmap")
    for line in fin:
        fout.write(line.replace(',', '.').replace(";;;;;;;;","").replace(";;;;;;;",""))
    fin.close()
    fout.close()
    file.append(open("%s_%s_data_mod.csv"%(versuchsname,i), encoding="charmap"))

beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=";"))
    
""" Konstanten """
S = [0.959, 0.995]
err_S = [0.022, 0.016]

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
#xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    #xerr.append([0.02 for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([0.01 for j in range(s[i])])
    
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    #xerr.append([0.02 for j in range(s[i])])
    y_data.append(np.array([((beamData[i][j,1])/S[i]) for j in range(s[i])]))
    yerr.append([((err_S[i]*beamData[i][j,1])/(S[i]**2)) + ((0.01)/(S[i])) for j in range(s[i])])

""" Plotten """
fig = []
ax = []
for i in range(2*countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
ax[0].set_title("Enfriamiento")
ax[1].set_title("Calentificacion")
ax[2].set_title("Enfriamiento")
ax[3].set_title("Calentificacion")

for i in range(countFiles):
    ax[i].set_xlabel("$t$ (s)")
    ax[i].set_ylabel("$V$ (mV)")
    
for i in range(countFiles):
    ax[i+2].set_xlabel("$t$ (s)")
    ax[i+2].set_ylabel("$\dot q$ (W)")

""" Regressionskurve """ 
x_data_unlimited = []
x_data_select = []
y_data_select = []
s1  = [753,209]
s2  = [1006,1033]
for i in range(countFiles):
    mask = [((a<=s1[i]) or (a>=s2[i])) for a in x_data[i]]
    x_data_unlimited.append(np.arange(0,x_data[i+2][-1],2))
    x_data_select.append(x_data[i+2][mask])
    y_data_select.append(y_data[i+2][mask])

def fitCurve(x, A, B, C, D):
    return (A * (np.asarray(x)**3)) + (B * (np.asarray(x)**2)) + (C * np.asarray(x)) + D


perr = []
fitRes = []
pFit = []
pCov = []
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data_select[i], y_data_select[i], p0=[0.5,0.5,-1,0]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(countFiles):
    A = fitRes[i][0][0].round(9)
    B = fitRes[i][0][1].round(6)
    C = fitRes[i][0][2].round(3)
    D = fitRes[i][0][3].round(2)
    fitCurveSym = A * (x**3) + B * (x**2) + C * x + D 
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    string = "Linia base: %s"%fitCurveSym
    ax[i+2].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), label=string, linewidth=2, color="blue")
    ax[i+2].legend()
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Weitere Plots """
y_data_new = []
x_data_new_limited = []
y_data_new_limited = []
Q = [0,0]
for i in range(countFiles):
    mask = [~((a<=s1[i]) or (a>=s2[i])) for a in x_data[i]]
    y_data_new.append(y_data[i]-fitCurve(x_data[i], *pFit[i]))
    y_data_new_limited.append(y_data_new[i][mask])
    x_data_new_limited.append(x_data[i+2][mask])
# =============================================================================
#     for a in y_data_new_limited[i]:
#         Q[i] += a;
#     Q = Q - 0.5* (y_data_new_limited[i][0] + y_data_new_limited[i][-1])
# =============================================================================
    for j in range(1,y_data_new_limited[i].size):
        Q[i] += 0.5 * (x_data_new_limited[i][j] - x_data_new_limited[i][j-1]) * (y_data_new_limited[i][j] + y_data_new_limited[i][j-1])
    fig.append(plt.figure())
    ax.append(fig[i+4].add_axes([0.15,0.15,0.75,0.75]))
    string = "$Q$ = %s J"%Q[i].round(3)
    ax[i+4].plot(x_data[i],y_data_new[i],'o',label=string,markersize=2,color="Black")
    ax[i+4].legend()
    ax[i+4].grid(True)
    
ax[4].set_title("Enfriamiento")
ax[5].set_title("Calentificacion")

for i in range(2*countFiles, 3*countFiles):
    ax[i].set_xlabel("$t$ (s)")
    ax[i].set_ylabel("$\dot q$ (W)")
    
""" Daten in CSV schreiben """
for i in range(countFiles):
    f = open("CSV_saves/%s_%s_table.csv"%(versuchsname,ersterDatenIndex+i), "w")
    writer = csv.writer(f)
    l = [x_data_new_limited[i], y_data_new_limited[i]]
    z = zip(*l)
    for a in z:
        writer.writerow(a)
    f.close
    
""" Daten in Tabelle ausgeben """
# =============================================================================
# b = []
# b.append(np.around(T,3))
# b.append(np.around(I,3))
# b.append(np.around(Y,3))
# b.append(np.around(S,3))
# labels = ["$T$ (Â°C)","$I(T)$ (A)","$Y(T)$ (mV)","$S(T)$ (mV/W)"]
# b = np.array(b).T
# fig.append(plt.figure())
# ax.append(fig[12].add_axes([0,0,1,1]))
# ax[12].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
# ax[12].axis("off")
# ax[12]
# =============================================================================


""" Plot speichern """
for i in range((3*countFiles)):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



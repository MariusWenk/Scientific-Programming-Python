# Marius Wenk, Pau GranÃ© i Claudi Vall

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 3
versuchsname = "P1"
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


""" Daten vorbereiten """
s1 = [100,100,100]
s2 = [350,500,300]

s = []
x_data = []
y_data = []
#xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:s1[i],0].size)
    s.append(beamData[i][s1[i]:s2[i],0].size)
    s.append(beamData[i][s2[i]:,0].size)
    x_data.append(beamData[i][:s1[i],0])
    x_data.append(beamData[i][s1[i]:s2[i],0])
    x_data.append(beamData[i][s2[i]:,0])
    #xerr.append([0.02 for j in range(s[i])])
    y_data.append(beamData[i][:s1[i],1])
    y_data.append(beamData[i][s1[i]:s2[i],1])
    y_data.append(beamData[i][s2[i]:,1])
    yerr.append([0.01 for j in range(s[(3*i)])])
    yerr.append([0.01 for j in range(s[(3*i)+1])])
    yerr.append([0.01 for j in range(s[(3*i)+2])])

""" Plotten """
fig = []
ax = []
for i in range(3*countFiles):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],fmt='o',markersize=2,color="Black")
    # ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
for i in range(3):
    ax[(3*i)].set_title("$T$ = 0°C, Mesura %s"%(i+1))
    ax[(3*i)+1].set_title("$T$ = 53°C, Mesura %s"%(i+1))
    ax[(3*i)+2].set_title("$T$ = 100°C, Mesura %s"%(i+1))
    
for i in range(3*countFiles):
    ax[i].set_xlabel("$t$ (s)")
    ax[i].set_ylabel("$V$ (mV)")

""" Mittelwerte """ 
y_data_select = []
s3 = [185,230,170]
s4 = [240,440,260]
s5 = [900,1420,810]
s6 = [970,1520,850]
for i in range(countFiles):
    y_data_select.append(y_data[(3*i)])
    mask = np.logical_and(beamData[i][:,0] >= s3[i], beamData[i][:,0] <= s4[i])
    y_data_select.append(beamData[i][mask,1])
    mask = np.logical_and(beamData[i][:,0] >= s5[i], beamData[i][:,0] <= s6[i])
    y_data_select.append(beamData[i][mask,1])
    
epsilons = []
for i in range(3*countFiles):
    if(i%3 == 0):
        epsilons.append(np.mean(y_data_select[i]))
    if(i%3 == 1):
        epsilons.append(np.mean(y_data_select[i+1]))
    if(i%3 == 2):
        epsilons.append(np.mean(y_data_select[i-1]))
    
mean_epsilons = []
for i in range(countFiles):
    mean_epsilons.append((epsilons[0+i] + epsilons[3+i] + epsilons[6+i])/3)
    
""" Weitere Plots """
T = [0, 53, 100]
eps_err = [0.01, 0.01, 0.01]
fig.append(plt.figure())
ax.append(fig[9].add_axes([0.15,0.15,0.75,0.75]))
ax[9].errorbar(T,mean_epsilons,eps_err,fmt='X',markersize=10,color="Black")
# ax[9].legend()
ax[9].grid(True)
# ax[9].axis([0,1,2,3])
# ax[9].set(xlim=(0,8))
# ax[9].set(ylim=(-0.2,2.2))
    
ax[9].set_xlabel("$t = T - T_{0}$ (K)")
ax[9].set_ylabel("$\epsilon$ (mV)")

""" Regressionskurve """
def fitCurve(x, A, B):
    return (A * (np.asarray(x)**2)) + (B * np.asarray(x))

perr = []
fitRes = []
pFit = []
pCov = []
for i in range(1):
    fitRes.append(curve_fit(fitCurve, T, mean_epsilons, p0=[0.5,-1]))
    pFit.append(fitRes[i][0])
    pCov.append(fitRes[i][1])
    perr.append(np.sqrt(np.diag(pCov[i])))
    print("Fitfehler",i+1,perr[i])
    #ax[i].set(xlim=(295,370))

""" Regressionsfunktion """
t = sp.symbols('t')
x_data_unlimited = []
for i in range(1):
    x_data_unlimited.append(np.arange(0,T[-1],0.5))
    A = fitRes[i][0][0].round(10)
    B = fitRes[i][0][1].round(5)
    fitCurveSym = B * t + A * (t**2)
    print("Regressionskurve %s: %s"%(i+1,fitCurveSym))
    string = "Función de calibración: %s"%fitCurveSym
    ax[i+9].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit[i]), label=string, linewidth=2, color="blue")
    ax[i+9].legend()
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
    
""" Daten in Tabelle ausgeben """
b = []
b.append(np.around(T,3))
b.append(np.around(epsilons[:3],3))
b.append(np.around(epsilons[3:6],3))
b.append(np.around(epsilons[6:],3))
b.append(np.around(mean_epsilons,3))
b.append(np.around(eps_err,3))
labels = ["$T$ (°C)","Mesura 1","Mesura 2","Mesura 3", "Valores medios", "Errores"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[10].add_axes([0,0,1,1]))
ax[10].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[10].axis("off")
ax[10]


""" Plot speichern """
for i in range((3*countFiles)+2):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,ersterDatenIndex+i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert



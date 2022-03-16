# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 20
versuchsname = "V2"
versuchsindex = 4
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH1.csv", encoding="charmap"))
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
err = [0.01,0.02,0.02]
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,0])
    xerr.append([err[i//3] for j in range(s[i])])
    y_data.append(beamData[i][:,1])
    yerr.append([2 for j in range(s[i])])

""" Daten vorbereiten """
countFiles = 2
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(1):
    s.append(len(f))
    x_data.append(1/I**2)
    xerr.append([del_I/I[j]**3 for j in range(s[i])])
    y_data.append(f)
    yerr.append(f_err)
    
for i in range(1):
    s.append(len(I))
    x_data.append(I)
    xerr.append([del_I for j in range(s[i])])
    y_data.append(the)
    yerr.append([del_the for j in range(s[i])])
    
print(x_data)
print(xerr)

""" Plotten """
versuchsname = "ELO2"

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
    
ax[0].set_xlabel("$1/I_L^2$ in 1/$A^2$")
ax[0].set_ylabel("$f$ in mm")
ax[1].set_xlabel("$I_L$ in A")
ax[1].set_ylabel("$\Theta$ in mm")


""" Regressionskurve """ 
xmax = [1.4,2.4]
xmin = [-0.1,0]
x_data_unlimited = []
for i in range(countFiles):
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B):
    return A * np.asarray(x) + B

fitRes = []
perr = []
x = sp.symbols('x')
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1, 1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    A = fitRes[i][0][0].round(2)
    B = fitRes[i][0][1].round(2)
    fitCurveStr = A * x + B
    string = "$f_1$(x) = %s"%fitCurveStr
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))
    
def fitCurve(x, A):
    return A * np.asarray(x)

fitRes = []
perr = []
x = sp.symbols('x')
for i in range(1):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    A = fitRes[i][0][0].round(2)
    fitCurveStr = A * x
    string = "$f_2$(x) = %s"%fitCurveStr
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./2.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert


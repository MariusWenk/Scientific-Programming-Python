# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp


""" Plotten """
versuchsname = "Laser1"

x_data = []
y_data = []
xerr = []
yerr = []

OD = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,1,2,3,4])
x_data.append(1/(10**OD))
y_data.append([463,390.8,308.4,243.3,187.2,145.4,111.4,45.3,4.7,0.4,0.1])
s = len(x_data[0])
xerr.append([0 for j in range(s)])
yerr.append([1 for j in range(s)])

x_data.append([2,5,10,15,20,25,30,35,40,45,50])
y_data.append([2.95,3.5,4.38,5.3,6.2,7.1,8,8.91,9.8,10.7,11.2])
s = len(x_data[1])
xerr.append([1 for j in range(s)])
yerr.append([0.01 for j in range(s)])

fig = []
ax = []
for i in range(2):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))

ax[0].set_xlabel("Transmissivität $T$")
ax[0].set_ylabel("$I_D$ in $\mu$A")
ax[1].set_xlabel("$x$ in mm")
ax[1].set_ylabel("$U_P$ in V")
    
""" Regressionskurve """ 
xmin = [0,0]
xmax = [1.1,52]
x_data_unlimited = []
for i in range(2):
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B):
    return A * np.asarray(x) + B

fitRes = []
perr = []
x = sp.symbols('x')
for i in range(2):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1, 1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    A = fitRes[i][0][0].round(2)
    B = fitRes[i][0][1].round(2)
    fitCurveStr = A * x + B
    string = "f(x) = %s"%fitCurveStr
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))
    
""" Plot speichern """
for i in range(2):
    fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
   
""" Tabelle ausgeben """
fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(OD,1))
    b.append(np.around(np.array(x_data[0])*100,2))
    b.append(np.around(y_data[0],1))
    labels = ["Filterstärke in OD","$T$ in %", "$I_D$ in $\mu$A"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    


# =============================================================================
# """ Daten auslesen """
# countFiles = 20
# versuchsindex = 4
# ersterDatenIndex = 1
# beamData1 = []
# beamData2 = []
# for i in range(countFiles):
#     i += ersterDatenIndex
#     file1 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH1.CSV", encoding="charmap")
#     file2 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH2.CSV", encoding="charmap")
#     beamData1.append(np.loadtxt(file1, delimiter=",", usecols=(3,4)))
#     beamData2.append(np.loadtxt(file2, delimiter=",", usecols=(3,4)))
#     file1.close()
#     file2.close()
#     
# """ Konstanten """
# U_x_err = 0.1
# I_err = 0.1
# 
# 
# """ Daten vorbereiten """
# s = []
# t_1 = []
# t_2 = []
# U_x = []
# I = []
# x_data = []
# y_data = []
# xerr = []
# yerr = []
# for i in range(countFiles):
#     # t_1.append(beamData1[i][:,0])
#     # t_2.append(beamData2[i][:,0])
#     U_x.append(beamData1[i][:,1])
#     I.append(beamData2[i][:,1])
#     s1 = len(U_x)
#     s2 = len(I)
#     size = min(s1,s2)
#     x_data_single = []
#     y_data_single = []
#     x_max = -1000
#     for j in range(size):
#         if U_x[i][j] in x_data_single:
#             x_max = U_x[i][j]
#             x_data_single.append(U_x[i][j])
#             y_data_single.append(I[i][j])
#     s.append(len(x_data_single))
#     x_data.append(x_data_single)
#     xerr.append([U_x_err for j in range(s[i])])
#     y_data.append(y_data_single)
#     yerr.append([I_err for j in range(s[i])])
# 
# """ Plotten """
# versuchsname = "Laser2"
# 
# fig = []
# ax = []
# for i in range(countFiles):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i].set_xlabel("$x$ in mm")
#     ax[i].set_ylabel("Intensität $U_T$ in V")
# =============================================================================


# =============================================================================
# """ Regressionskurve """ 
# xmax = [1.4,2.4]
# xmin = [-0.1,0]
# x_data_unlimited = []
# for i in range(countFiles):
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B):
#     return A * np.asarray(x) + B
# 
# fitRes = []
# perr = []
# x = sp.symbols('x')
# for i in range(countFiles):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1, 1]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     A = fitRes[i][0][0].round(2)
#     B = fitRes[i][0][1].round(2)
#     fitCurveStr = A * x + B
#     string = "$f_1$(x) = %s"%fitCurveStr
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     perr.append(np.sqrt(np.diag(pCov)))
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
#     
# def fitCurve(x, A):
#     return A * np.asarray(x)
# 
# fitRes = []
# perr = []
# x = sp.symbols('x')
# for i in range(1):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     A = fitRes[i][0][0].round(2)
#     fitCurveStr = A * x
#     string = "$f_2$(x) = %s"%fitCurveStr
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     perr.append(np.sqrt(np.diag(pCov)))
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
# 
# """ Plot speichern """
# for i in range(countFiles):
#     fig[i].savefig("./2.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
# 
# =============================================================================

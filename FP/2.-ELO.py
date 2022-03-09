# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

# =============================================================================
# """ Daten auslesen """
# countFiles = 8
# versuchsname = "V2"
# ersterDatenIndex = 1
# file = []
# for i in range(countFiles):
#     i += ersterDatenIndex
#     file.append(open("2.Daten/%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))
# beamData = []
# for i in range(countFiles):
#     beamData.append(np.loadtxt(file[i], delimiter=","))
#     
# """ Konstanten """
# A = -328.104
# B = 730.294
# A_err = 24.589
# B_err = 8.678
# 
# """ Daten vorbereiten """
# s = []
# x_data = []
# y_data = []
# xerr = []
# yerr = []
# err = [0.01,0.02,0.02]
# for i in range(countFiles):
#     s.append(beamData[i][:,0].size)
#     x_data.append(beamData[i][:,0])
#     xerr.append([err[i//3] for j in range(s[i])])
#     y_data.append(beamData[i][:,1])
#     yerr.append([2 for j in range(s[i])])
#     
# x_data_old = []
# xerr_old = []
# for i in range(3,countFiles):
#     x_data_old.append(x_data[i])
#     x_data[i] = [(A + B * x) for x in x_data[i]]
#     xerr_old.append(xerr[i])
#     xerr[i] = [(A_err + (B_err * x) + (B * 0.02)) for x in x_data_old[i-3]]
# 
# x_data[3] = x_data[3][::-1]
# y_data[3] = y_data[3][::-1]
# 
# """ Plotten """
# versuchsname = "ELO1"
# 
# fig = []
# ax = []
# stri = ["$I_H$","$U_W$","$U_W$"]
# val = ["2,20A","2,51A","2,70A","0,00V","0,00V","3,01V","6,00V","1,50V"]
# xlab = ["$U_W$ in V","$T$ in K","$T$ in K"]
# for i in range(countFiles):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i].set_xlabel(f"{xlab[i//3]}")
#     ax[i].set_ylabel("$I_S$ in $\mu$A")
#     ax[i].set_title(f"{stri[i//3]} = {val[i]}, {i//4+1}. Kathode")
#     
# fig.append(plt.figure())
# ax.append(fig[countFiles].add_axes([0.15,0.15,0.75,0.75]))
# for i in range(3):
#     ax[countFiles].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label=f"{stri[i//3]} = {val[i]}, {i//4+1}. Kathode",fmt='o',markersize=2)
#     ax[countFiles].legend()
#     ax[countFiles].grid(True)
# ax[countFiles].set_xlabel(f"{xlab[0]}")
# ax[countFiles].set_ylabel("$I_S$ in $\mu$A")
# 
# """ Regressionskurve """ 
# xmax = [1700,1700,1730,1730]
# xmin = [1100,1450,1450,1450]
# indize = [4,10,8,7]
# fitplots = [3,4,5,7]
# x_data_unlimited = []
# for i in range(4):
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B):
#     return A * np.asarray(x-B)
# 
# fitRes = []
# perr = []
# for i in range(4):
#     fitRes.append(curve_fit(fitCurve, x_data[fitplots[i]][indize[i]:], y_data[fitplots[i]][indize[i]:], p0=[-1, 1]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     A = fitRes[i][0][0].round(2)
#     B = fitRes[i][0][1].round(2)
#     string = f"f(x) = {A}*(x-{B})"
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[fitplots[i]].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[fitplots[i]].legend()
#     perr.append(np.sqrt(np.diag(pCov)))
#     print("Fitfehler",i+1,perr[i])
#     ax[fitplots[i]].set(xlim=(xmin[i],xmax[i]))
# 
#     
# """ Plotten """
# x_data_new = []
# xerr_new = []
# y_data_new = []
# yerr_new = []
# for i in range(4):
#     x_data[fitplots[i]] = np.array(x_data[fitplots[i]])
#     y_data[fitplots[i]] = np.array(y_data[fitplots[i]])
#     x_data_new.append(1/x_data[fitplots[i]])
#     xerr_new.append(xerr[fitplots[i]]/(x_data[fitplots[i]]**2))
#     y_data_new.append(np.log(y_data[fitplots[i]]/(x_data[fitplots[i]]**2)))
#     yerr_new.append((1/x_data[fitplots[i]]) + 2/(y_data[fitplots[i]]**2))
#     fig.append(plt.figure())
#     ax.append(fig[i+countFiles+1].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i+countFiles+1].errorbar(x_data_new[i],y_data_new[i],yerr_new[i],xerr_new[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i+countFiles+1].legend()
#     ax[i+countFiles+1].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i+countFiles+1].set_xlabel("$1/T$ in $K^{-1}$")
#     ax[i+countFiles+1].set_ylabel("$ln(I_S/T^2)$")
#     ax[i+countFiles+1].set_title(f"{stri[fitplots[i]//3]} = {val[fitplots[i]]}, {fitplots[i]//4+1}. Kathode")
# 
# """ Regressionskurve """ 
# xmax = [0.0009,0.0007,0.0007,0.0007]
# xmin = [0.00055,0.00055,0.00055,0.00055]
# x_data_unlimited = []
# for i in range(4):
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.00001))
# 
# def fitCurve(x, A, B):
#     return A * np.asarray(x) + B
# 
# fitRes = []
# perr = []
# minind = [1,1,0,0]
# x = sp.symbols('x')
# for i in range(4):
#     fitRes.append(curve_fit(fitCurve, x_data_new[i][minind[i]:indize[i]], y_data_new[i][minind[i]:indize[i]], p0=[-2000, 5]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     A = fitRes[i][0][0].round(2)
#     B = fitRes[i][0][1].round(2)
#     fitCurveStr = A * x + B
#     string = "f(x) = %s"%fitCurveStr
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i+countFiles+1].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i+countFiles+1].legend()
#     perr.append(np.sqrt(np.diag(pCov)))
#     print("Fitfehler",i+1,perr[i])
#     ax[i+countFiles+1].set(xlim=(xmin[i],xmax[i]))
#     
# """ Plot speichern """
# for i in range(countFiles+5):
#     fig[i].savefig("./2.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     
# """ Daten in Tabelle ausgeben """
# fig = []
# ax = []
# for i in range(3,countFiles):
#     b = []
#     b.append(np.around(x_data_old[i-3],3))
#     b.append(np.around(xerr_old[i-3],3))
#     b.append(np.around(x_data[i],3))
#     b.append(np.around(xerr[i],3))
#     b.append(np.around(y_data[i],3))
#     b.append(np.around(yerr[i],3))
#     labels = ["$I_H$ in A","$\Delta I_H$ in A",f"{xlab[i//3]}",f"$\Delta${xlab[i//3]}","$I_S$ in $\mu$A","$\Delta I_S$ in $\mu$A"]
#     b = np.array(b).T
#     fig.append(plt.figure())
#     ax.append(fig[i-3].add_axes([0,0,1,1]))
#     ax[i-3].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i-3].axis("off")
#     ax[i-3].set_title(f"{stri[i//3]} = {val[i]}, {i//4+1}. Kathode")
#     
# for i in range(4):
#     b = []
#     b.append(np.around(x_data_new[i][minind[i]:],6))
#     b.append(np.around(xerr_new[i][minind[i]:],6))
#     b.append(np.around(y_data_new[i][minind[i]:],3))
#     b.append(np.around(yerr_new[i][minind[i]:],3))
#     labels = ["$1/T$ in $K^{-1}$","$\Delta 1/T$ in $K^{-1}$","$ln(I_S/T^2)$","$\Delta ln(I_S/T^2)$"]
#     b = np.array(b).T
#     fig.append(plt.figure())
#     ax.append(fig[i+5].add_axes([0,0,1,1]))
#     ax[i+5].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i+5].axis("off")
#     ax[i+5].set_title(f"{stri[fitplots[i]//3]} = {val[fitplots[i]]}, {fitplots[i]//4+1}. Kathode")
# 
# """ Plot speichern """
# for i in range(9):
#     fig[i].savefig("./2.Plots/%s_%s_table.png"%(versuchsname,i), bbox_inches='tight', dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     
# =============================================================================
        

# =============================================================================
# """ Konstanten, Rechnungen """
# v_s = [4.011, 4.046, 4.723]
# v_s_err = 0.292
# p = [1.789, 2.484, 13.361]
# p_err = [(1+v_s_err*(62.3+p[i])+1.4*v_s[i])/(v_s[i]-1) for i in range(3)]
# print(p_err)
# del_s = 1
# del_l = 0.7
# del_v = 10.971
# v = [93.867, 100.406, 60.779]
# eps = [0.012, 0.0112, 0.0186]
# l = 217.5
# eps_err = [(del_s*+p_err[i]*(1+eps[i]*v[i])+del_l*eps[i]*v[i]+del_v*eps[i]*(l-p[i]))/(v[i]*(l-p[i]))for i in range(3)]
# print(eps_err)
# delta_err = [2*(p_err[i]+del_l*eps[i]+l*eps_err[i]) for i in range(3)]
# print(delta_err)
# f = [((l-p[i])*eps[i])/(1-eps[i]**2) for i in range(3)]
# print(f)
# f_err = [(del_l*eps[i]+p_err[i]*eps[i]+(l-p[i])*eps_err[i]) for i in range(3)]
# print(f_err)
# I = np.array([2.26, 1.56, 0.86])
# del_I = 0.01
# the = [45.86,1.75,-52.62]
# del_the = 2
# 
# """ Daten vorbereiten """
# countFiles = 2
# s = []
# x_data = []
# y_data = []
# xerr = []
# yerr = []
# for i in range(1):
#     s.append(len(f))
#     x_data.append(1/I**2)
#     xerr.append([del_I/I[j]**3 for j in range(s[i])])
#     y_data.append(f)
#     yerr.append(f_err)
#     
# for i in range(1):
#     s.append(len(I))
#     x_data.append(I)
#     xerr.append([del_I for j in range(s[i])])
#     y_data.append(the)
#     yerr.append([del_the for j in range(s[i])])
#     
# print(x_data)
# print(xerr)
# 
# """ Plotten """
# versuchsname = "ELO2"
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
#     
# ax[0].set_xlabel("$1/I_L^2$ in 1/$A^2$")
# ax[0].set_ylabel("$f$ in mm")
# ax[1].set_xlabel("$I_L$ in A")
# ax[1].set_ylabel("$\Theta$ in mm")
# 
# 
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


""" Konstanten, Rechnungen """
versuchsname = "ELO3"

dr = []
dr.append([14.27, 23.87])
dr.append([12.41, 13.51, 19.16, 23.51])
dr.append([7.76, 14.95, 26.67])
dr.append(["(-> 9.)", 38.38, 42.89, 48.03, 51.62, 54.72, 56.76, 60.19])

dl = []
dl.append([14.14, 21.54])
dl.append([12.31, 13.69, 19.61, 23.24])
dl.append([7.83, 14.58, 25.54])
dl.append([30.01, 37.25, 42.3, 47.24, 50.79, 54.82, 58.02, 61.37])

roff = [0,0,0,1]
b = 207.2
del_b = 2.1
del_r = 0.5

d = []
r = []
theta = []
theta_err = []
sintq = []
sintq_err = []
for i in range(4):
    d.append(np.array([(dr[i][j]+dl[i][j])/2 if j>=roff[i] else dl[i][j] for j in range(len(dr[i]))]))
    r.append(d[i]/2)
    theta.append(np.arctan(r[i]/b)/2)
    theta_err.append((1/(2*(1+((r[i]/b)**2))))*((del_r/b)+((del_b*r[i])/(b**2))))
    sintq.append(np.sin(theta[i])**2)
    sintq_err.append(2*np.sin(theta[i])*np.cos(theta[i])*theta_err[i])
    
""" Daten in Tabelle ausgeben """
stri = ["Thalliumchlorid", "Aluminium", "Molybdänoxid"]

fig = []
ax = []
for i in range(3):
    b = []
    if i == 1:
        dr[i] = dr[i] + dr[3]
        dl[i] = dl[i] + dl[3]
        d[i] = d[i].tolist() + d[3].tolist()
        r[i] = r[i].tolist() + r[3].tolist()
        theta[i] = theta[i].tolist() + theta[3].tolist()
        theta_err[i] = theta_err[i].tolist() + theta_err[3].tolist()
        sintq[i] = sintq[i].tolist() + sintq[3].tolist()
        sintq_err[i] = sintq_err[i].tolist() + sintq_err[3].tolist()
    b.append(dr[i])
    b.append(dl[i])
    b.append(np.around(d[i],3))
    b.append(np.around(r[i],3))
    b.append(np.around(theta[i],4))
    b.append(np.around(theta_err[i],4))
    b.append(np.around(sintq[i],5))
    b.append(np.around(sintq_err[i],5))
    labels = ["$d_{R}$ in mm","$d_{L}$ in mm","$d$ in mm","$r$ in mm","$\Theta$","$\Delta \Theta$","$sin^2(\Theta)$","$\Delta sin^2(\Theta)$"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    ax[i].set_title(f"{stri[i]}")
    
""" Plot speichern """
for i in range(3):
    fig[i].savefig("./2.Plots/%s_%s_table.png"%(versuchsname,stri[i]), bbox_inches='tight', dpi=100) # Bild als png Datei in Ordner Plots gespeichert


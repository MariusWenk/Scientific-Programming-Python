# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
from scipy.signal import find_peaks

# =============================================================================
# 
# """ Plotten """
# versuchsname = "Laser1"
# 
# x_data = []
# y_data = []
# xerr = []
# yerr = []
# 
# OD = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,1,2,3,4])
# x_data.append(1/(10**OD))
# y_data.append([463,390.8,308.4,243.3,187.2,145.4,111.4,45.3,4.7,0.4,0.1])
# s = len(x_data[0])
# xerr.append([0 for j in range(s)])
# yerr.append([1 for j in range(s)])
# 
# x_data.append([2.95,3.5,4.38,5.3,6.2,7.1,8,8.91,9.8,10.7,11.2])
# y_data.append([2,5,10,15,20,25,30,35,40,45,50])
# s = len(x_data[1])
# xerr.append([0.01 for j in range(s)])
# yerr.append([1 for j in range(s)])
# 
# fig = []
# ax = []
# for i in range(2):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
# 
# ax[0].set_xlabel("Transmissivität $T$")
# ax[0].set_ylabel("$I_D$ in $\mu$A")
# ax[1].set_xlabel("$U_P$ in V")
# ax[1].set_ylabel("$x$ in mm")
#     
# """ Regressionskurve """ 
# xmin = [0,1]
# xmax = [1.1,14]
# x_data_unlimited = []
# for i in range(2):
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B):
#     return A * np.asarray(x) + B
# 
# fitRes = []
# perr = []
# x = sp.symbols('a')
# for i in range(2):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1, 1]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     A = fitRes[i][0][0].round(2)
#     B = fitRes[i][0][1].round(2)
#     fitCurveStr = A * x + B
#     string = "f(a) = %s"%fitCurveStr
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     perr.append(np.sqrt(np.diag(pCov)))
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
#     
# """ Plot speichern """
# for i in range(2):
#     fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#    
# """ Tabelle ausgeben """
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     b.append(np.around(OD,1))
#     b.append(np.around(np.array(x_data[0])*100,2))
#     b.append(np.around(y_data[0],1))
#     labels = ["Filterstärke in OD","$T$ in %", "$I_D$ in $\mu$A"]
#     b = np.array(b).T        
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# 
# 
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
# d = [51+(3*i) for i in range(20)]
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
# A1 = fitRes[1][0][0]
# B1 = fitRes[1][0][1]
# A2 = perr[1][0]
# B2 = perr[1][1]
# for i in range(countFiles):
#     # t_1.append(beamData1[i][:,0])
#     # t_2.append(beamData2[i][:,0])
#     U_x.append(beamData1[i][:,1])
#     I.append(beamData2[i][:,1])
#     s1 = len(U_x[i])
#     s2 = len(I[i])
#     size = min(s1,s2)
#     x_data_single = []
#     y_data_single = []
#     for j in range(size):
#         if not U_x[i][j] in x_data_single:
#             x_data_single.append(U_x[i][j])
#             y_data_single.append(I[i][j])
#         n = x_data_single.index(min(x_data_single))
#     x_data_single.pop(n)
#     y_data_single.pop(n)
#     x_data_single = (A1 * np.array(x_data_single)) + B1
#     s.append(len(x_data_single))
#     x_data.append(x_data_single)
#     xerr.append([(A1 * U_x_err) + (A2 * x_data_single[j]) + B2 for j in range(s[i])])
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
#     ax[i].set_title(f"d = {d[i]}cm, sphärisch-sphärisch")
# 
# 
# """ Regressionskurve """ 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(countFiles):
#     xmax.append(max(x_data[i]) + 5)   
#     xmin.append(min(x_data[i]) - 5)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B, C):
#     return A * np.asarray(np.exp(-2*((x-C)**2)/(B**2)))
# 
# fitRes = []
# perr = []
# x = sp.symbols('x')
# for i in range(countFiles):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[0.4, 5, 14.5]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     perr.append(np.sqrt(np.diag(pCov)))
#     A = fitRes[i][0][0].round(3)
#     B = abs(fitRes[i][0][1].round(2))
#     C = fitRes[i][0][2].round(2)
#     string = "$f_{%s}$(x) = %s * exp(-(2$(x-%s)^2$)/$%s^2$)"%(i+1,A,C,B)
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
# 
# """ Plot speichern """
# for i in range(countFiles):
#     fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Tabelle ausgeben """
# versuchsname = "Strahlbreite_sp-sp"
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     w = np.array([abs(fitRes[j][0][1]) for j in range(len(d))])
#     w_err = np.array([perr[j][1] for j in range(len(d))])
#     b.append(np.around(d,1))
#     b.append(np.around([fitRes[j][0][0] for j in range(len(d))],3))
#     b.append(np.around([perr[j][0] for j in range(len(d))],3))
#     b.append(np.around(w,2))
#     b.append(np.around(w_err,3))
#     b.append(np.around([fitRes[j][0][2] for j in range(len(d))],2))
#     b.append(np.around([perr[j][2] for j in range(len(d))],3))
#     labels = ["$d$ in cm", "$I_0$ in V", "$\Delta I_0$ in V", "$w$ in mm", "$\Delta w$ in mm", "$x_0$ in mm", "$\Delta x_0$ in mm"]
#     b = np.array(b).T        
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Werte, Rechnungen """
# zs2 = 127.4
# as2 = 3
# zPT = -230
# del_a = 4.6
# lamb = 632.8e-7
# R = 60
# d = np.array(d)
# a = np.array([zs2 - (d[i]/2) - as2 - zPT for i in range(countFiles)])
# w = w * 0.1
# theta = np.arctan(w/a) * 1000
# theta_err = (1/(1+((w/a)**2)))*((w_err/a)+((del_a*w)/(a**2))) * 1000
# theta_theo = np.arctan((np.sqrt((2*lamb)/np.pi)*(1/np.power((d*(2*R-d)),(1/4))))) * 1000
# theta_diff = (abs(theta_theo-theta)/(theta_theo))*100
# print(sum(theta_diff)/len(theta_diff))
#     
# """ Tabelle ausgeben """
# versuchsname = "Winkeldivergenz_sp-sp"
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     b.append(np.around(d,1))
#     b.append(np.around(a,1))
#     b.append(np.around(theta,3))
#     b.append(np.around(theta_err,3))
#     b.append(np.around(theta_theo,3))
#     b.append(np.around(theta_diff,2))
#     labels = ["$d$ in cm", "$a$ in cm", "$\Theta$ in mrad", "$\Delta \Theta$ in mrad", "$\Theta_{theo}$ in mrad", "Abweichung in %"]
#     b = np.array(b).T        
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Plotten """
# x_data = [d]
# xerr = [[0.4 for i in range(len(d))]]
# y_data = [theta]
# yerr = [theta_err]
# fig = []
# ax = []
# for i in range(1):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i].set_xlabel("$d$ in cm")
#     ax[i].set_ylabel("Winkeldivergenz $\Theta$ in mrad")
#     ax[i].set_title("sphärisch-sphärisch")
#     
# def fitCurve(d):
#     return np.arctan(np.sqrt((2*lamb)/np.pi)*(1/np.power((d*(2*R-d)),(1/4)))) * 1000
# 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(1):
#     xmax.append(max(x_data[i]) + 3)   
#     xmin.append(min(x_data[i]) - 3)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i]), label="Theoriekurve",linewidth=2)
#     ax[i].legend()
# 
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
#     
#     
# """ Daten auslesen """
# countFiles = 20
# versuchsindex = 4
# ersterDatenIndex = 81
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
# d = [51,54,57,60,63,52,53,55,56,58,59,61,62,64,51.5,52.5,53.5,54.5,55.5,56.5]
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
#     s1 = len(U_x[i])
#     s2 = len(I[i])
#     size = min(s1,s2)
#     x_data_single = []
#     y_data_single = []
#     for j in range(size):
#         if not U_x[i][j] in x_data_single:
#             x_data_single.append(U_x[i][j])
#             y_data_single.append(I[i][j])
#         n = x_data_single.index(min(x_data_single))
#     x_data_single.pop(n)
#     y_data_single.pop(n)
#     x_data_single = (A1 * np.array(x_data_single)) + B1
#     s.append(len(x_data_single))
#     x_data.append(x_data_single)
#     xerr.append([(A1 * U_x_err) + (A2 * x_data_single[j]) + B2 for j in range(s[i])])
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
#     ax[i].set_title(f"d = {d[i]}cm, sphärisch-plan")
# 
# 
# """ Regressionskurve """ 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(countFiles):
#     xmax.append(max(x_data[i]) + 5)   
#     xmin.append(min(x_data[i]) - 5)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
# 
# def fitCurve(x, A, B, C):
#     return A * np.asarray(np.exp(-2*((x-C)**2)/(B**2)))
# 
# fitRes = []
# perr = []
# x = sp.symbols('x')
# for i in range(countFiles):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[0.4, 5, 14.5]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     perr.append(np.sqrt(np.diag(pCov)))
#     A = fitRes[i][0][0].round(3)
#     B = abs(fitRes[i][0][1].round(2))
#     C = fitRes[i][0][2].round(2)
#     string = "$f_{%s}$(x) = %s * exp(-(2$(x-%s)^2$)/$%s^2$)"%(i+1,A,C,B)
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
#     ax[i].legend()
#     print("Fitfehler",i+1,perr[i])
#     ax[i].set(xlim=(xmin[i],xmax[i]))
# 
# """ Plot speichern """
# for i in range(countFiles):
#     fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Tabelle ausgeben """
# versuchsname = "Strahlbreite_sp-pl"
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     w = np.array([abs(fitRes[j][0][1]) for j in range(len(d))])
#     w_err = np.array([perr[j][1] for j in range(len(d))])
#     b.append(np.around(d,1))
#     b.append(np.around([fitRes[j][0][0] for j in range(len(d))],3))
#     b.append(np.around([perr[j][0] for j in range(len(d))],3))
#     b.append(np.around(w,2))
#     b.append(np.around(w_err,3))
#     b.append(np.around([fitRes[j][0][2] for j in range(len(d))],2))
#     b.append(np.around([perr[j][2] for j in range(len(d))],3))
#     labels = ["$d$ in cm", "$I_0$ in V", "$\Delta I_0$ in V", "$w$ in mm", "$\Delta w$ in mm", "$x_0$ in mm", "$\Delta x_0$ in mm"]
#     b = np.array(b).T
#     b = b[b[:, 0].argsort()]
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Werte, Rechnungen """
# zs2 = 140.6
# as2 = 3.4
# zPT = -171
# del_a = 4.6
# lamb = 632.8e-7
# R = 75
# d = np.array(d)
# a = np.array([zs2 - (d[i]/2) - as2 - zPT for i in range(countFiles)])
# w = w * 0.1
# theta = np.arctan(w/a) * 1000
# theta_err = (1/(1+((w/a)**2)))*((w_err/a)+((del_a*w)/(a**2))) * 1000
# theta_theo = np.arctan((np.sqrt((lamb)/np.pi)*(1/np.power((d*(R-d)),(1/4))))) * 1000
# theta_diff = (abs(theta_theo-theta)/(theta_theo))*100
# print(sum(theta_diff)/len(theta_diff))
#     
# """ Tabelle ausgeben """
# versuchsname = "Winkeldivergenz_sp-pl"
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     b.append(np.around(d,1))
#     b.append(np.around(a,1))
#     b.append(np.around(theta,3))
#     b.append(np.around(theta_err,3))
#     b.append(np.around(theta_theo,3))
#     b.append(np.around(theta_diff,2))
#     labels = ["$d$ in cm", "$a$ in cm", "$\Theta$ in mrad", "$\Delta \Theta$ in mrad", "$\Theta_{theo}$ in mrad", "Abweichung in %"]
#     b = np.array(b).T
#     b = b[b[:, 0].argsort()]
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Plotten """
# x_data = [d]
# xerr = [[0.4 for i in range(len(d))]]
# y_data = [theta]
# yerr = [theta_err]
# fig = []
# ax = []
# for i in range(1):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i].set_xlabel("$d$ in cm")
#     ax[i].set_ylabel("Winkeldivergenz $\Theta$ in mrad")
#     ax[i].set_title("sphärisch-plan")
#     
# def fitCurve(d):
#     return np.arctan(np.sqrt((lamb)/np.pi)*(1/np.power((d*(R-d)),(1/4)))) * 1000
# 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(1):
#     xmax.append(max(x_data[i]) + 1)   
#     xmin.append(min(x_data[i]) - 1)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i]), label="Theoriekurve",linewidth=2)
#     ax[i].legend()
# 
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
# =============================================================================


# =============================================================================
# """ Daten vorbereiten """
# d = np.array([51+(3*i) for i in range(20)])
# del_d = 0.4
# I_D = np.array([130.7,123.5,119.8,119.7,123.3,132.1,109.5,112.2,116.1,123,111.8,103.7,102.8,110.7,85.2,80.2,80.5,82.2,66,25.5])
# del_I_D = 1
# del_I_D1 = 10.47
# R = 0.735e-1
# del_R = 0.01e-1
# lamb = 632.8e-7
# F = (R**2)/(lamb*d)
# F_err = ((R**2)/(lamb*(d**2)) * del_d) + ((R*2*del_R)/(lamb*d))
# P = (1.95/476.45)*I_D
# P_err = (((1.95*del_I_D1)/(476.45**2))*I_D) + ((1.95/476.45)*del_I_D)
# 
# """ Plotten """
# versuchsname = "Ausgangsleistung"
# 
# x_data = [F]
# xerr = [F_err]
# y_data = [P]
# yerr = [P_err]
# fig = []
# ax = []
# for i in range(1):
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
#     ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
#     ax[i].legend()
#     ax[i].grid(True)
#     # ax[i].axis([0,1,2,3])
#     # ax[i].set(xlim=(0,8))
#     # ax[i].set(ylim=(-0.2,2.2))
#     ax[i].set_xlabel("Fresnelzahl $F$")
#     ax[i].set_ylabel("$P$ in mW")
#     ax[i].set_title("sphärisch-sphärisch")
#     
# """ Regressionskurve """ 
# xmax = []
# xmin = []
# x_data_unlimited = []
# for i in range(1):
#     xmax.append(max(x_data[i]) + 0.05)   
#     xmin.append(min(x_data[i]) - 0.05)
#     x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.001))
# 
# def fitCurve(x, A, B, C):
#     return A - B * np.asarray(np.exp(C/(x**1.5)))
# 
# fitRes = []
# perr = []
# for i in range(1):
#     fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[4,2.5,0.19]))
#     pFit = fitRes[i][0]
#     pCov = fitRes[i][1]
#     perr.append(np.sqrt(np.diag(pCov)))
#     A = pFit[0].round(4)
#     B = pFit[1].round(4)
#     C = pFit[2].round(4)
#     string = "$P(F)$ = %s - %s * exp(%s*$F^{-(3/2)}$)"%(A,B,C)
#     print("Regressionskurve %s: %s"%(i+1,string))
#     ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string, linewidth=2)
#     ax[i].legend()
#     print("Fitfehler",i+1,np.around(perr[i],4))
#     ax[i].set(xlim=(xmin[i],xmax[i]))
# 
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
#     
# """ Tabelle ausgeben """
# fig = []
# ax = []
# for i in range(1):
#     b = []
#     b.append(np.around(d,1))
#     b.append(np.around(I_D,1))
#     b.append(np.around(P,3))
#     b.append(np.around(P_err,3))
#     b.append(np.around(F,3))
#     b.append(np.around(F_err,3))
#     labels = ["$d$ in cm", "$I_D$ in $\mu$A", "$P$ in mW", "$\Delta P$ in mW", "Fresnelzahl $F$", "$\Delta F$"]
#     b = np.array(b).T
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     
# """ Plot speichern """
# for i in range(1):
#     fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     plt.close(fig[i])
# =============================================================================
    

""" Daten auslesen """
countFiles = 60
versuchsindex = 4
ersterDatenIndex = 21
beamData = []
for i in range(countFiles):
    i += ersterDatenIndex
    file = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH1.CSV", encoding="charmap")
    beamData.append(np.loadtxt(file, delimiter=",", usecols=(3,4)))
    file.close()
    
""" Konstanten """
d = [51+(3*i) for i in range(20)]
U_err = 0.001
t_err = [0.0001,0.00001,0.00001]

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(np.array(beamData[i][:,0]))
    xerr.append([t_err[i%3] for j in range(s[i])])
    y_data.append(np.array(beamData[i][:,1]))
    yerr.append([U_err for j in range(s[i])])
    
""" Plotten """
versuchsname = "Longitudinalmoden"
stri = ["zu $\Delta T$", "zu $\Delta t$", "mit Nachleuchten"]
invested = [2,2,2,2,2,3,2,3,3,3,2,2,2,3,3,3,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,1,2]
invested_indizes = []

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
    ax[i].set_xlabel("$t$ in s")
    ax[i].set_ylabel("$U_F$ in V")
    ax[i].set_title(f"$d$ = {d[i//3]}cm, Aufnahme {stri[i%3]}")
    
    if i%3 == 0 or i%3 == 1:
        y_max1, _ = find_peaks(y_data[i], height=0.02)
        ax[i].plot(x_data[i][y_max1],y_data[i][y_max1],'o',label="Maxima",markersize=10,color="Blue")
    
        y_max2, _ = find_peaks(y_data[i][y_max1], height=0.02)
        ax[i].plot(x_data[i][y_max1][y_max2],y_data[i][y_max1][y_max2],'o',label="Maxima",markersize=10,color="Green")
        
        y_max3, _ = find_peaks(y_data[i][y_max1][y_max2], height=0.02)
        ax[i].plot(x_data[i][y_max1][y_max2][y_max3],y_data[i][y_max1][y_max2][y_max3],'o',label="Maxima",markersize=10,color="Red")
    
        if invested[((i//3)*2)+(i%3)] == 1:
            invested_indizes.append(y_max1)
        elif invested[((i//3)*2)+(i%3)] == 2:
            invested_indizes.append(y_max1[y_max2])
        else:
            invested_indizes.append(y_max1[y_max2[y_max3]])

        
""" Regressionskurve """ 
xmax = []
xmin = []
x_data_unlimited = []
for i in range(countFiles//3):
    xmax.append(max(x_data[(i*3)+2]) + 0.001)   
    xmin.append(min(x_data[(i*3)+2]) - 0.001)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.0001))

def fitCurve(x, A, B, C):
    return A * np.asarray(np.exp(-((x-C)**2)/((B**2)*2)))

fitRes = []
perr = []
for i in range(countFiles//3):
    fitRes.append(curve_fit(fitCurve, x_data[(i*3)+2], y_data[(i*3)+2], p0=[0.4, 0.02, 0.1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    perr.append(np.sqrt(np.diag(pCov)))
    A = fitRes[i][0][0].round(4)
    B = abs(fitRes[i][0][1].round(4))
    C = fitRes[i][0][2].round(4)
    string = "$f_{%s}$(x) = %s * exp(-($(x-%s)^2$)/2*$%s^2$)"%(i+1,A,C,B)
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[(i*3)+2].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[(i*3)+2].legend()
    print("Fitfehler",i+1,perr[i])
    ax[(i*3)+2].set(xlim=(xmin[i],xmax[i]))

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Tabelle ausgeben """
versuchsname = "Verstärkungsbreite_FPI"

fig = []
ax = []

g = 2.35482
for i in range(1):
    b = []
    w = np.array([abs(fitRes[j][0][1]) for j in range(len(d))])*1000
    w_err = np.array([perr[j][1] for j in range(len(d))])*1000
    fwhm = w*g
    fwhm_err = w_err*g
    b.append(np.around(d,1))
    b.append(np.around(np.array([fitRes[j][0][0] for j in range(len(d))])*1000,2))
    b.append(np.around(np.array([perr[j][0] for j in range(len(d))])*1000,2))
    b.append(np.around(w,3))
    b.append(np.around(w_err,3))
    b.append(np.around(np.array([fitRes[j][0][2] for j in range(len(d))])*1000,3))
    b.append(np.around(np.array([perr[j][2] for j in range(len(d))])*1000,3))
    b.append(np.around(fwhm,3))
    b.append(np.around(fwhm_err,3))
    labels = ["$d$ in cm", "$I_0$ in mV", "$\Delta I_0$ in mV", "$\sigma$ in ms", "$\Delta \sigma$ in ms", "$\mu$ in ms", "$\Delta \mu$ in ms", "$F$ in ms", "$\Delta F$ in ms"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
print(sum(fwhm)/len(fwhm))
print(sum(fwhm_err)/len(fwhm_err))

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    

""" Tabelle ausgeben """
versuchsname = "Longitudinalmoden"

fig = []
ax = []
for i in range((countFiles//3)*2):
    b = []
    b.append(np.arange(invested_indizes[i].size).astype(int))
    b.append(np.around(x_data[((i//2)*3)+(i%2)][invested_indizes[i]],6))
    b.append(np.around(y_data[((i//2)*3)+(i%2)][invested_indizes[i]],6))
    labels = ["#", "$t$ in s", "$U_F$ in V"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")

""" Plot speichern """
for i in range((countFiles//3)*2):
    fig[i].savefig("./4.Plots/%s_%s_table.png"%(versuchsname,((i//2)*3)+(i%2)+ersterDatenIndex), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
versuchsname = "Maxima_FPI"
    
elected = []
elected.append([0,1,3,4])
elected.append([3,8,4])
elected.append([0,4,7,9])
elected.append([3,12,5])
elected.append([0,3,5,7])
elected.append([0,4,4])
elected.append([2,4,6,7])
elected.append([0,4,5])
elected.append([0,1,2,3])
elected.append([2,5,4])
elected.append([1,4,7,9])
elected.append([3,8,4]) #5
elected.append([1,2,4,6])
elected.append([1,3,3])
elected.append([0,1,2,3])
elected.append([1,3,3])
elected.append([2,3,8,13])
elected.append([2,6,4])
elected.append([0,2,3,4])
elected.append([0,3,4])
elected.append([1,2,3,4])
elected.append([3,10,3])
elected.append([0,2,3,4])
elected.append([1,3,3])
elected.append([0,3,6,10])
elected.append([6,12,4])
elected.append([0,2,5,6])
elected.append([3,7,4])
elected.append([0,1,2,4])
elected.append([0,3,4])
elected.append([1,2,3,4])
elected.append([1,5,4])
elected.append([0,2,5,6])
elected.append([1,5,3])
elected.append([0,1,4,7])
elected.append([1,3,3])
elected.append([0,1,2,3])
elected.append([0,4,4])
elected.append([0,2,6,7])
elected.append([0,1,2])

DeltaT1 = []
DeltaT2 = []
Deltatn = []
n = []
for i in range(countFiles//3):
    DeltaT1.append(x_data[(i*3)+0][invested_indizes[(i*2)+0]][elected[(i*2)+0]][1] - x_data[(i*3)+0][invested_indizes[(i*2)+0]][elected[(i*2)+0]][0])
    DeltaT2.append(x_data[(i*3)+0][invested_indizes[(i*2)+0]][elected[(i*2)+0]][3] - x_data[(i*3)+0][invested_indizes[(i*2)+0]][elected[(i*2)+0]][2])
    Deltatn.append(x_data[(i*3)+1][invested_indizes[(i*2)+1]][elected[(i*2)+1][:2]][1] - x_data[(i*3)+1][invested_indizes[(i*2)+1]][elected[(i*2)+1][:2]][0])
    n.append(elected[(i*2)+1][2]-1)

DeltaT1 = np.array(DeltaT1)
DeltaT2 = np.array(DeltaT2)
Deltatn = np.array(Deltatn)
n = np.array(n)
Deltat = Deltatn/n
DeltaT = (DeltaT1 + DeltaT2) /2

""" Tabelle ausgeben """
fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(d,1))
    b.append(np.around(DeltaT1*1000,3))
    b.append(np.around(DeltaT2*1000,3))
    b.append(np.around(DeltaT*1000,3))
    b.append(np.around(Deltatn*1000,3))
    b.append(np.around(n,0))
    b.append(np.around(Deltat*1000,3))
    labels = ["$d$ in cm", "$\Delta T_1$ in ms", "$\Delta T_2$ in ms", "$\Delta T$ in ms", "$\Delta t * n$ in ms", "$n$", "$\Delta t$ in ms"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
    print(sum(DeltaT)/len(DeltaT))

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Tabelle ausgeben """
versuchsname = "Modenfrequenzen_original"

delt = 0.0001/n
delT = 0.0002
c = 299792458
d_err = 0.4
d = np.array(d)
Deltanu = (10*Deltat)/DeltaT
Deltanu_err = ((delt*10)/DeltaT) + ((Deltat*10*delT)/(DeltaT**2))
DeltanuT = (c/(2*d*1e-2))*1e-9
diff = ((abs(Deltanu-DeltanuT))/DeltanuT)*100

fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(d,1))
    b.append(np.around(Deltanu*1000,2))
    b.append(np.around(Deltanu_err*1000,2))
    b.append(np.around(DeltanuT*1000,2))
    b.append(np.around(diff,2))
    labels = ["$d$ in cm", "$\Delta ν_L$ in MHz", "$\Delta \Delta ν_L$ in MHz", "$\Delta ν_T$ in MHz", "Abweichung in %"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Tabelle ausgeben """
versuchsname = "Modenfrequenzen_angepasst"

for i in range(5):
    Deltanu[i] = Deltanu[i]*2

diff = ((abs(Deltanu-DeltanuT))/DeltanuT)*100

print((sum(diff)/len(diff)))
    
fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(d,1))
    b.append(np.around(Deltanu*1000,2))
    b.append(np.around(Deltanu_err*1000,2))
    b.append(np.around(DeltanuT*1000,2))
    b.append(np.around(diff,2))
    labels = ["$d$ in cm", "$\Delta ν_L$ in MHz", "$\Delta \Delta ν_L$ in MHz", "$\Delta ν_T$ in MHz", "Abweichung in %"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Plotten """
x_data = []
y_data = []
xerr = []
yerr = []

x_data.append(1/d)
y_data.append(Deltanu)
s = len(x_data[0])
xerr.append(d_err/(d**2))
yerr.append(Deltanu_err)

fig = []
ax = []
for i in range(1):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel("$1/d$ in 1/cm")
    ax[i].set_ylabel("$\Delta ν_L$ in GHz")
    
""" Regressionskurve """ 
xmax = []
xmin = []
x_data_unlimited = []
for i in range(1):
    xmax.append(max(x_data[i]) + 0.002)
    xmin.append(min(x_data[i]) - 0.002)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.001))

def fitCurve(x, A):
    return A * np.asarray(x)

fitRes = []
perr = []
x = sp.symbols('a')
for i in range(1):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    A = fitRes[i][0][0].round(2)
    fitCurveStr = A * x
    string = "f(a) = %s"%fitCurveStr
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
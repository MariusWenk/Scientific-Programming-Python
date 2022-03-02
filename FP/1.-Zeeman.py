# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
import more_itertools as mi
from scipy.optimize import curve_fit
import sympy as sp

""" Daten auslesen """
countFiles = 24
file = []
farbArray = ["A","B","C"]
for i in range(countFiles):
    if i<12:
        pol = 0
        messung = 12 + i//3
    else:
        pol = 90
        messung = 8 + i//3
    farbe = farbArray[i%3]
    file.append(open("1.Daten/%s%s%s.csv"%(messung,farbe,pol), encoding="charmap"))
beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))
    
""" Konstanten """
t = 2.499
t_err = 0.002
B = [0.833,0.770,0.592,0.405]

""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,5])
    y_data.append(beamData[i][:,6])
    
r = []
square_r_err = []
for i in range(countFiles):
    x0 = x_data[i][0]
    y0 = y_data[i][0]
    le = x_data[i].size
    if i<12:
        messung = 12 + i//3
    else:
        messung = 8 + i//3
    if messung < 14:
        delx0 = 3
        dely0 = 3
        delrk = 3.7
    else:
        delx0 = 2
        dely0 = 2
        delrk = 2.16
    pre_r = []
    pre_r_square_err = []
    for j in range(1,le):
        pre_r.append(np.sqrt(((x_data[i][j]-x0)**2)+((y_data[i][j]-y0)**2)))
        pre_r_square_err.append((2*abs(x_data[i][j]-x0)*delx0)+(2*abs(y_data[i][j]-y0)*dely0)+delrk)
    r.append(np.array(pre_r))
    square_r_err.append(np.array(pre_r_square_err))
    
square_r = np.array(r)**2

square = []
square_cut = []
Delta = []
delta = []
for i in range(countFiles//2):
    ma = min(square_r[i][::2].size,square_r[i+countFiles//2].size,square_r[i][1::2].size)
    square.append([square_r[i][::2],square_r[i+countFiles//2],square_r[i][1::2]])
    Delta.append([a[1:] - a[:-1] for a in [square_r[i][::2],square_r[i+countFiles//2],square_r[i][1::2]]])
    square_cut.append(np.array([square_r[i][::2][:ma],square_r[i+countFiles//2][:ma],square_r[i][1::2][:ma]]))
    delta.append(square_cut[i][1:,:] - square_cut[i][:-1,:])
    
square_err = []
square_cut_err = []
Delta_err = []
delta_err = []
for i in range(countFiles//2):
    ma = min(square_r_err[i][::2].size,square_r_err[i+countFiles//2].size,square_r_err[i][1::2].size)
    square_err.append([square_r_err[i][::2],square_r_err[i+countFiles//2],square_r_err[i][1::2]])
    Delta_err.append([a[1:] + a[:-1] for a in [square_r_err[i][::2],square_r_err[i+countFiles//2],square_r_err[i][1::2]]])
    square_cut_err.append(np.array([square_r_err[i][::2][:ma],square_r_err[i+countFiles//2][:ma],square_r_err[i][1::2][:ma]]))
    delta_err.append(square_cut_err[i][1:,:] + square_cut_err[i][:-1,:])
    
""" Mittelung """
Delta_med = []
delta_ab_med = []
delta_bc_med = []
delta_med = []
Delta_med_err = []
delta_ab_med_err = []
delta_bc_med_err = []
delta_med_err = []
ny_ab = []
ny_bc = []
ny = []
ny_ab_err = []
ny_bc_err = []
ny_err = []
for i in range(countFiles//2):
    pre_Delta_med = np.around((np.sum(Delta[i][0][::2])+np.sum(Delta[i][1][::2])+np.sum(Delta[i][2][::2]))/(Delta[i][0][::2].size+Delta[i][1][::2].size+Delta[i][2][::2].size),3)
    Delta_med.append(pre_Delta_med)
    pre_delta_ab_med = np.around((np.sum(delta[i][0]))/(delta[i][0].size),3)
    delta_ab_med.append(pre_delta_ab_med)
    pre_delta_bc_med = np.around((np.sum(delta[i][1]))/(delta[i][1].size),3)
    delta_bc_med.append(pre_delta_bc_med)
    pre_delta_med = np.around((np.sum(delta[i][0])+np.sum(delta[i][1]))/(delta[i][0].size+delta[i][1].size),3)
    delta_med.append(pre_delta_med)
    
    pre_Delta_med_err = np.around((np.sum(Delta_err[i][0][::2])+np.sum(Delta_err[i][1][::2])+np.sum(Delta_err[i][2][::2]))/(Delta_err[i][0][::2].size+Delta_err[i][1][::2].size+Delta_err[i][2][::2].size),3)
    Delta_med_err.append(pre_Delta_med_err)
    pre_delta_ab_med_err = np.around((np.sum(delta_err[i][0]))/(delta_err[i][0].size),3)
    delta_ab_med_err.append(pre_delta_ab_med_err)
    pre_delta_bc_med_err = np.around((np.sum(delta_err[i][1]))/(delta_err[i][1].size),3)
    delta_bc_med_err.append(pre_delta_bc_med_err)
    pre_delta_med_err = np.around((np.sum(delta_err[i][0])+np.sum(delta_err[i][1]))/(delta_err[i][0].size+delta_err[i][1].size),3)
    delta_med_err.append(pre_delta_med_err)
    
    pre_ny_ab = np.around((pre_delta_ab_med/pre_Delta_med)*(1/(2*t)),3)
    ny_ab.append(pre_ny_ab)
    pre_ny_bc = np.around((pre_delta_bc_med/pre_Delta_med)*(1/(2*t)),3)
    ny_bc.append(pre_ny_bc)
    pre_ny = np.around((pre_delta_med/pre_Delta_med)*(1/(2*t)),3)
    ny.append(pre_ny)
    
    pre_ny_ab_err = np.around(((pre_delta_ab_med/pre_Delta_med)*(1/(2*t)) * ((pre_delta_ab_med_err/pre_delta_ab_med)+(pre_Delta_med_err/pre_Delta_med)+(t_err/t))),3)
    ny_ab_err.append(pre_ny_ab_err)
    pre_ny_bc_err = np.around(((pre_delta_bc_med/pre_Delta_med)*(1/(2*t)) * ((pre_delta_bc_med_err/pre_delta_bc_med)+(pre_Delta_med_err/pre_Delta_med)+(t_err/t))),3)
    ny_bc_err.append(pre_ny_bc_err)
    pre_ny_err = np.around(abs(((pre_delta_med/pre_Delta_med)*(1/(2*t)) * ((pre_delta_med_err/pre_delta_ab_med)+(pre_Delta_med_err/pre_Delta_med)+(t_err/t)))),3)
    ny_err.append(pre_ny_err)
    
    
""" Daten in Tabelle ausgeben """
fig = []
ax = []
for i in range(countFiles//2):
    b = []
    zeros = [["" for i in range(delta[i][0].size)],["" for i in range(delta[i][1].size)]]
    ma = max(square_r[i][::2].size,square_r[i+countFiles//2].size,square_r[i][::2].size)
    b.append(["" if i%2==1 else 1+(i//2) for i in range((ma*2)-1)])
    b.append(np.around(list(mi.roundrobin(square[i][0],Delta[i][0])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][0],Delta[i][0])))))
    b.append(list(mi.roundrobin(np.around(delta[i][0],3),zeros[0]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta[i][0],zeros[0])))))
    b.append(np.around(list(mi.roundrobin(square[i][1],Delta[i][1])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][1],Delta[i][1])))))
    b.append(list(mi.roundrobin(np.around(delta[i][1],3),zeros[1]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta[i][1],zeros[1])))))
    b.append(np.around(list(mi.roundrobin(square[i][2],Delta[i][2])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][2],Delta[i][2])))))
    labels = ["Ordnung","innerste","","mittlere","","äußerste"]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    ax[i]
    
for i in range(countFiles//2):
    b = []
    zeros = [["" for i in range(delta[i][0].size)],["" for i in range(delta[i][1].size)]]
    ma = max(square_r_err[i][::2].size,square_r_err[i+countFiles//2].size,square_r_err[i][::2].size)
    b.append(["" if i%2==1 else 1+(i//2) for i in range((ma*2)-1)])
    b.append(np.around(list(mi.roundrobin(square_err[i][0],Delta_err[i][0])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][0],Delta_err[i][0])))))
    b.append(list(mi.roundrobin(np.around(delta_err[i][0],3),zeros[0]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta_err[i][0],zeros[0])))))
    b.append(np.around(list(mi.roundrobin(square_err[i][1],Delta_err[i][1])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][1],Delta_err[i][1])))))
    b.append(list(mi.roundrobin(np.around(delta_err[i][1],3),zeros[1]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta_err[i][1],zeros[1])))))
    b.append(np.around(list(mi.roundrobin(square_err[i][2],Delta_err[i][2])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][2],Delta_err[i][2])))))
    labels = ["Ordnung","innerste","","mittlere","","äußerste"]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i+countFiles//2].add_axes([0,0,1,1]))
    ax[i+countFiles//2].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i+countFiles//2].axis("off")
    ax[i+countFiles//2]
    
labelz = []
table_names = []
for i in range(countFiles//2):
    if i<12:
        messung = 12 + i//3
    farbe = farbArray[i%3]
    labelz.append(f"{messung}{farbe}")
    
b = []
table_names.append("(capital)Delta")
b.append(labelz)
b.append(Delta_med)
b.append(Delta_med_err)
labels = ["Messung","$\Delta$","$\Delta$ $\Delta$"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[countFiles].add_axes([0,0,1,1]))
ax[countFiles].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[countFiles].axis("off")
ax[countFiles]

b = []
table_names.append("delta")
b.append(labelz)
b.append(delta_ab_med)
b.append(delta_ab_med_err)
b.append(delta_bc_med)
b.append(delta_bc_med_err)
b.append(delta_med)
b.append(delta_med_err)
labels = ["Messung","$\delta_{mi}$","$\Delta$ $\delta_{mi}$","$\delta_{äm}$","$\Delta$ $\delta_{äm}$","$\delta$","$\Delta$ $\delta$"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[countFiles+1].add_axes([0,0,1,1]))
ax[countFiles+1].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[countFiles+1].axis("off")
ax[countFiles+1]

b = []
table_names.append("Ny")
b.append(labelz)
b.append(ny_ab)
b.append(ny_ab_err)
b.append(ny_bc)
b.append(ny_bc_err)
b.append(ny)
b.append(ny_err)
labels = ["Messung","$ν_{mi}$","$\Delta$ $ν_{mi}$","$ν_{äm}$","$\Delta$ $ν_{äm}$","$ν$","$\Delta$ $ν$"]
b = np.array(b).T
fig.append(plt.figure())
ax.append(fig[countFiles+2].add_axes([0,0,1,1]))
ax[countFiles+2].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
ax[countFiles+2].axis("off")
ax[countFiles+2]

""" Plot speichern """
for i in range(countFiles//2):
    if i<12:
        messung = 12 + i//3
    farbe = farbArray[i%3]
    fig[i].savefig(f"./1.Plots/{messung}{farbe}_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    
for i in range(countFiles//2):
    if i<12:
        messung = 12 + i//3
    farbe = farbArray[i%3]
    fig[i+countFiles//2].savefig(f"./1.Plots/{messung}{farbe}_error_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert
for i in range(3):
    fig[i+countFiles].savefig(f"./1.Plots/{table_names[i]}_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    
    
""" Plotten """
versuchsname = "zeeman"

fig = []
ax = []
lambdas = ["546,1","435,8","405,7"]
xerr = [0.001 for i in range(len(B))]
for i in range(3):
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(B,ny[i::3],ny_err[i::3],xerr[i],label="Werte mit Fehler",fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    ax[i].set_xlabel("$B$ in T")
    ax[i].set_ylabel("$\Delta$ $ν$ in $mm^{-1}$")
    ax[i].set_title(f"$\lambda$ = {lambdas[i]}nm")

""" Regressionskurve """ 
x_data_unlimited = np.arange(0,0.9,0.01)

def fitCurve(x, A):
    return A * np.asarray(x)

fitRes = []
perr = []
for i in range(3):
    fitRes.append(curve_fit(fitCurve, B, ny[i::3], p0=[-1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
    ax[i].legend()
    perr.append(np.sqrt(np.diag(pCov)))
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(0,0.9))

""" Regressionsfunktion """
x = sp.symbols('x')
for i in range(3):
    A = fitRes[i][0][0].round(4)
    fitCurve = A * x
    print("Regressionskurve %s: %s"%(i+1,fitCurve))
    # lambdified_fitCurve = sp.lambdify(x,fitCurve)
    # #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))

""" Plot speichern """
for i in range(3):
    fig[i].savefig("./1.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
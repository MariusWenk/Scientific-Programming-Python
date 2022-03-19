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

x_data.append([2.95,3.5,4.38,5.3,6.2,7.1,8,8.91,9.8,10.7,11.2])
y_data.append([2,5,10,15,20,25,30,35,40,45,50])
s = len(x_data[1])
xerr.append([0.01 for j in range(s)])
yerr.append([1 for j in range(s)])

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
ax[1].set_xlabel("$U_P$ in V")
ax[1].set_ylabel("$x$ in mm")
    
""" Regressionskurve """ 
xmin = [0,1]
xmax = [1.1,14]
x_data_unlimited = []
for i in range(2):
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B):
    return A * np.asarray(x) + B

fitRes = []
perr = []
x = sp.symbols('a')
for i in range(2):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[-1, 1]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    A = fitRes[i][0][0].round(2)
    B = fitRes[i][0][1].round(2)
    fitCurveStr = A * x + B
    string = "f(a) = %s"%fitCurveStr
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
    


""" Daten auslesen """
countFiles = 20
versuchsindex = 4
ersterDatenIndex = 1
beamData1 = []
beamData2 = []
for i in range(countFiles):
    i += ersterDatenIndex
    file1 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH1.CSV", encoding="charmap")
    file2 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH2.CSV", encoding="charmap")
    beamData1.append(np.loadtxt(file1, delimiter=",", usecols=(3,4)))
    beamData2.append(np.loadtxt(file2, delimiter=",", usecols=(3,4)))
    file1.close()
    file2.close()
    
""" Konstanten """
U_x_err = 0.1
I_err = 0.1
d = [51+(3*i) for i in range(20)]


""" Daten vorbereiten """
s = []
t_1 = []
t_2 = []
U_x = []
I = []
x_data = []
y_data = []
xerr = []
yerr = []
A1 = fitRes[1][0][0]
B1 = fitRes[1][0][1]
A2 = perr[1][0]
B2 = perr[1][1]
for i in range(countFiles):
    # t_1.append(beamData1[i][:,0])
    # t_2.append(beamData2[i][:,0])
    U_x.append(beamData1[i][:,1])
    I.append(beamData2[i][:,1])
    s1 = len(U_x[i])
    s2 = len(I[i])
    size = min(s1,s2)
    x_data_single = []
    y_data_single = []
    for j in range(size):
        if not U_x[i][j] in x_data_single:
            x_data_single.append(U_x[i][j])
            y_data_single.append(I[i][j])
        n = x_data_single.index(min(x_data_single))
    x_data_single.pop(n)
    y_data_single.pop(n)
    x_data_single = (A1 * np.array(x_data_single)) + B1
    s.append(len(x_data_single))
    x_data.append(x_data_single)
    xerr.append([(A1 * U_x_err) + (A2 * x_data_single[j]) + B2 for j in range(s[i])])
    y_data.append(y_data_single)
    yerr.append([I_err for j in range(s[i])])

""" Plotten """
versuchsname = "Laser2"

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
    ax[i].set_xlabel("$x$ in mm")
    ax[i].set_ylabel("Intensität $U_T$ in V")
    ax[i].set_title(f"d = {d[i]}cm, sphährisch-sphärisch")


""" Regressionskurve """ 
xmax = []
xmin = []
x_data_unlimited = []
for i in range(countFiles):
    xmax.append(max(x_data[i]) + 5)   
    xmin.append(min(x_data[i]) - 5)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B, C):
    return A * np.asarray(np.exp(-2*((x-C)**2)/(B**2)))

fitRes = []
perr = []
x = sp.symbols('x')
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[0.4, 5, 14.5]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    perr.append(np.sqrt(np.diag(pCov)))
    A = fitRes[i][0][0].round(3)
    B = abs(fitRes[i][0][1].round(2))
    C = fitRes[i][0][2].round(2)
    string = "$f_{%s}$(x) = %s * exp(-(2$(x-%s)^2$)/$%s^2$)"%(i+1,A,C,B)
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Tabelle ausgeben """
versuchsname = "Strahlbreite_sp-sp"
fig = []
ax = []
for i in range(1):
    b = []
    w = np.array([abs(fitRes[j][0][1]) for j in range(len(d))])
    w_err = np.array([perr[j][1] for j in range(len(d))])
    b.append(np.around(d,1))
    b.append(np.around([fitRes[j][0][0] for j in range(len(d))],3))
    b.append(np.around([perr[j][0] for j in range(len(d))],3))
    b.append(np.around(w,2))
    b.append(np.around(w_err,3))
    b.append(np.around([fitRes[j][0][2] for j in range(len(d))],2))
    b.append(np.around([perr[j][2] for j in range(len(d))],3))
    labels = ["$d$ in cm", "$I_0$ in V", "$\Delta I_0$ in V", "$w$ in mm", "$\Delta w$ in mm", "$x_0$ in mm", "$\Delta x_0$ in mm"]
    b = np.array(b).T        
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Werte, Rechnungen """
zs2 = 127.4
as2 = 3
zPT = -230
del_a = 4.6
lamb = 632.8e-7
R = 60
d = np.array(d)
a = np.array([zs2 - (d[i]/2) - as2 - zPT for i in range(countFiles)])
w = w * 0.1
theta = np.arctan(w/a) * 1000
theta_err = (1/(1+((w/a)**2)))*((w_err/a)+((del_a*w)/(a**2))) * 1000
theta_theo = np.arctan((np.sqrt((2*lamb)/np.pi)*(1/np.power((d*(2*R-d)),(1/4))))) * 1000
theta_diff = (abs(theta_theo-theta)/(theta_theo))*100
print(sum(theta_diff)/len(theta_diff))
    
""" Tabelle ausgeben """
versuchsname = "Winkeldivergenz_sp-sp"
fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(d,1))
    b.append(np.around(a,1))
    b.append(np.around(theta,3))
    b.append(np.around(theta_err,3))
    b.append(np.around(theta_theo,3))
    b.append(np.around(theta_diff,2))
    labels = ["$d$ in cm", "$a$ in cm", "$\Theta$ in mrad", "$\Delta \Theta$ in mrad", "$\Theta_{theo}$ in mrad", "Abweichung in %"]
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
x_data = [d]
xerr = [[0.4 for i in range(len(d))]]
y_data = [theta]
yerr = [theta_err]
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
    ax[i].set_xlabel("$d$ in cm")
    ax[i].set_ylabel("Winkeldiergenz $\Theta$ in mrad")
    ax[i].set_title("sphährisch-sphärisch")
    
def fitCurve(d):
    return np.arctan(np.sqrt((2*lamb)/np.pi)*(1/np.power((d*(2*R-d)),(1/4)))) * 1000

xmax = []
xmin = []
x_data_unlimited = []
for i in range(1):
    xmax.append(max(x_data[i]) + 3)   
    xmin.append(min(x_data[i]) - 3)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i]), label="Theoriekurve",linewidth=2)
    ax[i].legend()

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
    
    
""" Daten auslesen """
countFiles = 20
versuchsindex = 4
ersterDatenIndex = 81
beamData1 = []
beamData2 = []
for i in range(countFiles):
    i += ersterDatenIndex
    file1 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH1.CSV", encoding="charmap")
    file2 = open(f"{versuchsindex}.Daten/ALL{i:04d}/F{i:04d}CH2.CSV", encoding="charmap")
    beamData1.append(np.loadtxt(file1, delimiter=",", usecols=(3,4)))
    beamData2.append(np.loadtxt(file2, delimiter=",", usecols=(3,4)))
    file1.close()
    file2.close()
    
""" Konstanten """
U_x_err = 0.1
I_err = 0.1
d = [51,54,57,60,63,52,53,55,56,58,59,61,62,64,51.5,52.5,53.5,54.5,55.5,56.5]


""" Daten vorbereiten """
s = []
t_1 = []
t_2 = []
U_x = []
I = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    # t_1.append(beamData1[i][:,0])
    # t_2.append(beamData2[i][:,0])
    U_x.append(beamData1[i][:,1])
    I.append(beamData2[i][:,1])
    s1 = len(U_x[i])
    s2 = len(I[i])
    size = min(s1,s2)
    x_data_single = []
    y_data_single = []
    for j in range(size):
        if not U_x[i][j] in x_data_single:
            x_data_single.append(U_x[i][j])
            y_data_single.append(I[i][j])
        n = x_data_single.index(min(x_data_single))
    x_data_single.pop(n)
    y_data_single.pop(n)
    x_data_single = (A1 * np.array(x_data_single)) + B1
    s.append(len(x_data_single))
    x_data.append(x_data_single)
    xerr.append([(A1 * U_x_err) + (A2 * x_data_single[j]) + B2 for j in range(s[i])])
    y_data.append(y_data_single)
    yerr.append([I_err for j in range(s[i])])

""" Plotten """
versuchsname = "Laser2"

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
    ax[i].set_xlabel("$x$ in mm")
    ax[i].set_ylabel("Intensität $U_T$ in V")
    ax[i].set_title(f"d = {d[i]}cm, sphährisch-plan")


""" Regressionskurve """ 
xmax = []
xmin = []
x_data_unlimited = []
for i in range(countFiles):
    xmax.append(max(x_data[i]) + 5)   
    xmin.append(min(x_data[i]) - 5)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))

def fitCurve(x, A, B, C):
    return A * np.asarray(np.exp(-2*((x-C)**2)/(B**2)))

fitRes = []
perr = []
x = sp.symbols('x')
for i in range(countFiles):
    fitRes.append(curve_fit(fitCurve, x_data[i], y_data[i], p0=[0.4, 5, 14.5]))
    pFit = fitRes[i][0]
    pCov = fitRes[i][1]
    perr.append(np.sqrt(np.diag(pCov)))
    A = fitRes[i][0][0].round(3)
    B = abs(fitRes[i][0][1].round(2))
    C = fitRes[i][0][2].round(2)
    string = "$f_{%s}$(x) = %s * exp(-(2$(x-%s)^2$)/$%s^2$)"%(i+1,A,C,B)
    print("Regressionskurve %s: %s"%(i+1,string))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i], *pFit), label=string,linewidth=2)
    ax[i].legend()
    print("Fitfehler",i+1,perr[i])
    ax[i].set(xlim=(xmin[i],xmax[i]))

""" Plot speichern """
for i in range(countFiles):
    fig[i].savefig("./4.Plots/%s_%s_plot.png"%(versuchsname,i), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Tabelle ausgeben """
versuchsname = "Strahlbreite_sp-pl"
fig = []
ax = []
for i in range(1):
    b = []
    w = np.array([abs(fitRes[j][0][1]) for j in range(len(d))])
    w_err = np.array([perr[j][1] for j in range(len(d))])
    b.append(np.around(d,1))
    b.append(np.around([fitRes[j][0][0] for j in range(len(d))],3))
    b.append(np.around([perr[j][0] for j in range(len(d))],3))
    b.append(np.around(w,2))
    b.append(np.around(w_err,3))
    b.append(np.around([fitRes[j][0][2] for j in range(len(d))],2))
    b.append(np.around([perr[j][2] for j in range(len(d))],3))
    labels = ["$d$ in cm", "$I_0$ in V", "$\Delta I_0$ in V", "$w$ in mm", "$\Delta w$ in mm", "$x_0$ in mm", "$\Delta x_0$ in mm"]
    b = np.array(b).T
    b = b[b[:, 0].argsort()]
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Werte, Rechnungen """
zs2 = 140.6
as2 = 3.4
zPT = -171
del_a = 4.6
lamb = 632.8e-7
R = 75
d = np.array(d)
a = np.array([zs2 - (d[i]/2) - as2 - zPT for i in range(countFiles)])
w = w * 0.1
theta = np.arctan(w/a) * 1000
theta_err = (1/(1+((w/a)**2)))*((w_err/a)+((del_a*w)/(a**2))) * 1000
theta_theo = np.arctan((np.sqrt((lamb)/np.pi)*(1/np.power((d*(R-d)),(1/4))))) * 1000
theta_diff = (abs(theta_theo-theta)/(theta_theo))*100
print(sum(theta_diff)/len(theta_diff))
    
""" Tabelle ausgeben """
versuchsname = "Winkeldivergenz_sp-pl"
fig = []
ax = []
for i in range(1):
    b = []
    b.append(np.around(d,1))
    b.append(np.around(a,1))
    b.append(np.around(theta,3))
    b.append(np.around(theta_err,3))
    b.append(np.around(theta_theo,3))
    b.append(np.around(theta_diff,2))
    labels = ["$d$ in cm", "$a$ in cm", "$\Theta$ in mrad", "$\Delta \Theta$ in mrad", "$\Theta_{theo}$ in mrad", "Abweichung in %"]
    b = np.array(b).T
    b = b[b[:, 0].argsort()]
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    
""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_table.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])
    
""" Plotten """
x_data = [d]
xerr = [[0.4 for i in range(len(d))]]
y_data = [theta]
yerr = [theta_err]
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
    ax[i].set_xlabel("$d$ in cm")
    ax[i].set_ylabel("Winkeldiergenz $\Theta$ in mrad")
    ax[i].set_title("sphährisch-plan")
    
def fitCurve(d):
    return np.arctan(np.sqrt((lamb)/np.pi)*(1/np.power((d*(R-d)),(1/4)))) * 1000

xmax = []
xmin = []
x_data_unlimited = []
for i in range(1):
    xmax.append(max(x_data[i]) + 1)   
    xmin.append(min(x_data[i]) - 1)
    x_data_unlimited.append(np.arange(xmin[i],xmax[i],0.01))
    ax[i].plot(x_data_unlimited[i], fitCurve(x_data_unlimited[i]), label="Theoriekurve",linewidth=2)
    ax[i].legend()

""" Plot speichern """
for i in range(1):
    fig[i].savefig("./4.Plots/%s_plot.png"%(versuchsname), dpi=100) # Bild als png Datei in Ordner Plots gespeichert
    plt.close(fig[i])


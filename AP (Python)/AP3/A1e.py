# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import math

""" Daten auslesen """
countFiles = 5
versuchsname = "A1e"
ersterDatenIndex = 1
file = []
for i in range(countFiles):
    i += ersterDatenIndex
    file.append(open("%s_%s_data.csv"%(versuchsname,i), encoding="charmap"))
beamData = []
for i in range(countFiles):
    beamData.append(np.loadtxt(file[i], delimiter=","))
""" Konstanten """
eta = 18.216e-6
ro_0 = 874
ro_L = 1.189
g = 9.81
d = 6e-3
del_d = 0.05e-3
del_U = 10
Us = [600,500,400,300]
A = 0.007776e-6
pi = 3.141592654

""" Daten vorbereiten """
s = []
v_F = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles+4):
    if (i == 0):
        s.append(beamData[i][:,0].size)
        v_F_single = (beamData[i][:,2]*0.27e-3)/beamData[i][:,1]
        del_v_F_single = ((0.0135e-3)/beamData[i][:,1])+(((beamData[i][:,2]*0.27e-3)/(beamData[i][:,1]**2))*0.25)
        x_data_single = np.sqrt((9*v_F_single*eta)/(2*(ro_0-ro_L)*g)) # = r
        xerr_single = (np.sqrt((9*eta)/(2*(ro_0-ro_L)*g)))*0.5*del_v_F_single/np.sqrt(v_F_single) # = del_r
        y_data_single = ((9*pi*d)/beamData[i][:,0])*np.sqrt((2*(eta**3)*(v_F_single**3))/((ro_0-ro_L)*g))
        yerr_single = (((9*pi*d)/(beamData[i][:,0]**2))*np.sqrt((2*(eta**3)*(v_F_single**3))/((ro_0-ro_L)*g))*del_U) + ((9*pi*d)/beamData[i][:,0])*np.sqrt((2*(eta**3)*(v_F_single*(del_v_F_single**2)))/((ro_0-ro_L)*g)) + (((9*pi)/beamData[i][:,0])*np.sqrt((2*(eta**3)*(v_F_single**3))/((ro_0-ro_L)*g))*del_d)
    elif (i <= 4 and i >= 1):
        s.append(beamData[i][:,0].size)
        v_E_single = (5*0.3e-3)/beamData[i][:,0]
        del_v_E_single = ((0.0135e-3)/beamData[i][:,0])+(((5*0.3e-3)/(beamData[i][:,0]**2))*0.01)
        v_F_single = (5*0.3e-3)/beamData[i][:,1]
        del_v_F_single = ((0.0135e-3)/beamData[i][:,1])+(((5*0.3e-3)/(beamData[i][:,1]**2))*0.01)
        x_data_single = np.sqrt((9*v_F_single*eta)/(2*(ro_0-ro_L)*g)) # = r
        xerr_single = (np.sqrt((9*eta)/(2*(ro_0-ro_L)*g)))*0.5*del_v_F_single/np.sqrt(v_F_single) # = del_r
        y_data_single = (((9*pi*d)/Us[i-1])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)
        yerr_single = ((((9*pi*d)/Us[i-1]**2)*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)*del_U) + ((((9*pi*d)/Us[i-1])*(((np.sqrt(((eta**3))/((ro_0-ro_L)*g*2*v_F_single)))*(v_F_single+v_E_single))+(np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g))))*del_v_F_single)) + ((((9*pi*d)/Us[i-1])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*del_v_E_single) + ((((9*pi)/Us[i-1])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)*del_d)
    else:
        s.append(beamData[i-4][:,0].size)
        v_E_single = (5*0.3e-3)/beamData[i-4][:,0]
        del_v_E_single = ((0.0135e-3)/beamData[i-4][:,0])+(((5*0.3e-3)/(beamData[i-4][:,0]**2))*0.01)
        v_F_single = (5*0.3e-3)/beamData[i-4][:,1]
        del_v_F_single = ((0.0135e-3)/beamData[i-4][:,1])+(((5*0.3e-3)/(beamData[i-4][:,1]**2))*0.01)
        x_data_single = np.sqrt((9*v_F_single*eta)/(2*(ro_0-ro_L)*g)) # = r
        xerr_single = (np.sqrt((9*eta)/(2*(ro_0-ro_L)*g)))*0.5*del_v_F_single/np.sqrt(v_F_single) # = del_r
        q_pre = (((9*pi*d)/Us[i-5])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)
        q_pre_err = ((((9*pi*d)/Us[i-5]**2)*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)*del_U) + ((((9*pi*d)/Us[i-5])*(((np.sqrt(((eta**3))/((ro_0-ro_L)*g*2*v_F_single)))*(v_F_single+v_E_single))+(np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g))))*del_v_F_single)) + ((((9*pi*d)/Us[i-5])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*del_v_E_single) + ((((9*pi)/Us[i-5])*np.sqrt((2*(eta**3)*(v_F_single))/((ro_0-ro_L)*g)))*(v_F_single+v_E_single)*del_d)
        y_data_single = q_pre/(1+(A/x_data_single))**1.5
        yerr_single = (q_pre_err/(1+(A/x_data_single))**1.5) + (q_pre*1.5*A*(xerr_single/((x_data_single**2)*(1+(A/x_data_single))**2.5)))
    v_F.append(v_F_single)
    x_data.append(x_data_single)
    xerr.append(xerr_single)
    y_data.append(y_data_single)
    yerr.append(yerr_single)

""" Plotten """
fig = []
ax = []
for i in range(countFiles+4):
    st = "Schwebemethode"
    if (i>0):
        st = "Gegenfeldmethode"
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0.15,0.15,0.75,0.75]))
    ax[i].errorbar(x_data[i],y_data[i],yerr[i],xerr[i],label="Werte für q nach %s"%st,fmt='o',markersize=2,color="Black")
    ax[i].legend()
    ax[i].grid(True)
    ax[i].set_xlabel("r in $m$")
    ax[i].set_ylabel("q in $C$")
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
for i in range(1,9):
    st = "ohne"
    if (i>5):
        st = "mit"
    ax[i].set_title("U = %s$V$ %s Cunningham-Korrektur"%(Us[(i-1)%4],st))
    
""" Daten in Tabelle ausgeben """
for i in range(countFiles):
    b = []
    if (i==0):
        b.append(np.around(beamData[i][:,0],2))
        b.append(np.around(x_data[i],10))
        b.append(np.around(xerr[i],10))
        b.append(np.around(y_data[i],22))
        b.append(np.around(yerr[i],22))
        labels = ["U in $V$","r in $m$","del r in $m$","$q_S$ in $C$","del $q_S$ in $C$"]
    else:
        b.append(np.around(x_data[i],10))
        b.append(np.around(xerr[i],10))
        b.append(np.around(y_data[i],22))
        b.append(np.around(yerr[i],22))
        b.append(np.around(y_data[i+4],22))
        b.append(np.around(yerr[i+4],22))
        labels = ["r in $m$","del r in $m$","$q_G$ in $C$","del $q_G$ in $C$","$q_C$ in $C$","del $q_C$ in $C$"]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i+countFiles+4].add_axes([0,0,1,1]))
    ax[i+countFiles+4].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i+countFiles+4].axis("off")
    ax[i+countFiles+4]
    
for i in range(10,14):
    ax[i].set_title("U = %s$V$"%(Us[(i-2)%4])) 
    
""" Histogram """
klassenbreite = 9e-20
balkenzahl = 32
y_histogramm = []
for i in range(countFiles):
    index = i
    if (i!=0):
        index += 4
    counts = []
    for j in range(balkenzahl):
        count = 0
        for q in y_data[index]:
            if (q>=(j*klassenbreite) and q<((j+1)*klassenbreite)):
                count += 1
        counts.append(count)
    y_histogramm.append(counts)
    x_data.append([(klassenbreite)*(j+0.5)/1e-19 for j in range(balkenzahl)])
    y_data.append(counts)
x_histogramm = []
for j in range(balkenzahl):
    if (j%4 == 0):
        x_histogramm.append(str(round((klassenbreite)*(j+0.5)/1e-19,2)))
    else:
        x_histogramm.append("")
x_pos = np.arange(balkenzahl)


for i in range(countFiles):
    fig.append(plt.figure())
    ax.append(fig[i+countFiles+9].add_axes([0.15,0.15,0.75,0.75]))
    ax[i+countFiles+9].bar(x_pos,y_histogramm[i],label="Histogramm",color="Black")
    ax[i+countFiles+9].set_xticks(range(balkenzahl))
    ax[i+countFiles+9].set_xticklabels(x_histogramm, rotation='vertical')
    ax[i+countFiles+9].legend()
    ax[i+countFiles+9].set_xlabel("q in 10^-19$C$")
    ax[i+countFiles+9].set_ylabel("Messanzahl N")
    # ax[i].axis([0,1,2,3])
    # ax[i].set(xlim=(0,8))
    # ax[i].set(ylim=(-0.2,2.2))
    
for i in range(14,19):
    st = "Schwebemethode"
    if (i>14):
        st = "Gegenfeldmethode mit U = %s$V$"%Us[(i-3)%4]
    ax[i].set_title(st)
                
        
        
""" Regressionskurve """ 
plots = 4
factor = 1
# inf = [j for j in range(balkenzahl) if j%3==0]
# sup = [j+3 for j in range(balkenzahl-3) if j%3==0]
inf = [0,7,13,18]
sup = [7,13,18,23]
x_data_limited = []
for j in range(plots):
    x_data_limited.append(x_data[countFiles+4][inf[j]*factor:sup[j]*factor])
y_data_limited = []
for i in range(countFiles):
    y_data_limited_single = []
    for j in range(plots):
        y_data_limited_single.append(y_data[i+countFiles+4][inf[j]*factor:sup[j]*factor])
    y_data_limited.append(y_data_limited_single)

x_data_unlimited = np.arange(0,29,0.01)

def fitCurve(x, A, B, C):
    return (1/np.sqrt(2*pi*B))* np.exp(-((x-A)**2)/(2*(B**2))) * C

def flatCurve(x):
    return 0*np.asarray(x)

perr = []
fitRes = []
for i in range(14,19):
    fitRes_single = []
    perr_single = []
    for j in range(plots):
        try:
            fitRes_single.append(curve_fit(fitCurve, x_data_limited[j], y_data_limited[i-14][j], p0=[-1,4,0.1], maxfev = 2000))
            pFit = fitRes_single[len(fitRes_single)-1][0]
            pCov = fitRes_single[len(fitRes_single)-1][1]
            ax[i].plot(x_data_unlimited, fitCurve(x_data_unlimited, *pFit), label="Fitkurve durch kleinste Quadrate",linewidth=2)
            perr_single.append(np.sqrt(np.diag(pCov)))
        except RuntimeError:
            print("Not possible")
        
    fitRes.append(fitRes_single)
    perr.append(perr_single)
    ax[i].legend()
#     print("Fitfehler",i+1,perr[i])

""" Regressionsfunktion """
# x = sp.symbols('x')
for i in range(5):
    for j in range(len(fitRes[i])):
        A = fitRes[i][j][0][0].round(3)
        delA = perr[i][j][0].round(3)
        B = fitRes[i][j][0][1].round(3)
        delB = perr[i][j][1].round(3)
        print("Erwartungswert Plot %s, Kurve %s = %s mit Fehler %s"%(i+1,j+1,A,delA))
        print("Standardabweichung Plot %s, Kurve %s = %s mit Fehler %s"%(i+1,j+1,B,delB))
#     fitCurve = A * x + B
#     print("Regressionskurve %s: %s"%(i+1,fitCurve))
#     lambdified_fitCurve = sp.lambdify(x,fitCurve)
    #Nulstellen:
    # print("Nullstellen %s: %s"%(i+1,np.roots(fitRes[i][0])))
    # maxFit = [fitRes[i][0][0]+perr[i][0],fitRes[i][0][1]-perr[i][1]]
    # print("Nullstellenfehler %s: %s"%(i+1,np.roots(maxFit)[0]-np.roots(fitRes[i][0])[0]))
    
""" Plot speichern """
for i in range(2*countFiles+9):
    fig[i].savefig("./Plots/%s_%s_plot.png"%(versuchsname,i+ersterDatenIndex), dpi=100,bbox_inches='tight') # Bild als png Datei in Ordner Plots gespeichert


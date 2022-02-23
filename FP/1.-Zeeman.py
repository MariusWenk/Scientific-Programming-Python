# Marius Wenk, Fernando Grumpe

""" Bibliotheken importieren """
import numpy as np
from matplotlib import pyplot as plt
import more_itertools as mi

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


""" Daten vorbereiten """
s = []
x_data = []
y_data = []
xerr = []
yerr = []
for i in range(countFiles):
    s.append(beamData[i][:,0].size)
    x_data.append(beamData[i][:,5])
    xerr.append([1 for j in range(s[i])])
    y_data.append(beamData[i][:,6])
    yerr.append([1 for j in range(s[i])])
    
r = []
for i in range(countFiles):
    x0 = x_data[i][0]
    y0 = y_data[i][0]
    le = x_data[i].size
    pre_r = []
    for j in range(1,le):
        pre_r.append(np.sqrt(((x_data[i][j]-x0)**2)+((y_data[i][j]-y0)**2)))
    r.append(np.array(pre_r))
    
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
    labels = ["","","","","",""]
    b = np.array(b).T
    fig.append(plt.figure())
    ax.append(fig[i].add_axes([0,0,1,1]))
    ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
    ax[i].axis("off")
    ax[i]

""" Plot speichern """
for i in range(countFiles//2):
    if i<12:
        messung = 12 + i//3
    else:
        messung = 8 + i//3
    farbe = farbArray[i%3]
    fig[i].savefig(f"./1.Plots/{messung}{farbe}_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert

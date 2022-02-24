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
t = 2.49
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
    pre_ny_err = np.around(((pre_delta_med/pre_Delta_med)*(1/(2*t)) * ((pre_delta_med_err/pre_delta_ab_med)+(pre_Delta_med_err/pre_Delta_med)+(t_err/t))),3)
    ny_err.append(pre_ny_err)
    
    print(np.around(pre_ny/B[i//3],3))
    
    
""" Daten in Tabelle ausgeben """
# =============================================================================
# fig = []
# ax = []
# for i in range(countFiles//2):
#     b = []
#     zeros = [["" for i in range(delta[i][0].size)],["" for i in range(delta[i][1].size)]]
#     ma = max(square_r[i][::2].size,square_r[i+countFiles//2].size,square_r[i][::2].size)
#     b.append(["" if i%2==1 else 1+(i//2) for i in range((ma*2)-1)])
#     b.append(np.around(list(mi.roundrobin(square[i][0],Delta[i][0])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][0],Delta[i][0])))))
#     b.append(list(mi.roundrobin(np.around(delta[i][0],3),zeros[0]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta[i][0],zeros[0])))))
#     b.append(np.around(list(mi.roundrobin(square[i][1],Delta[i][1])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][1],Delta[i][1])))))
#     b.append(list(mi.roundrobin(np.around(delta[i][1],3),zeros[1]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta[i][1],zeros[1])))))
#     b.append(np.around(list(mi.roundrobin(square[i][2],Delta[i][2])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square[i][2],Delta[i][2])))))
#     labels = ["Ordnung","innerste","","mittlere","","äußerste"]
#     b = np.array(b).T
#     fig.append(plt.figure())
#     ax.append(fig[i].add_axes([0,0,1,1]))
#     ax[i].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i].axis("off")
#     ax[i]
#     
# for i in range(countFiles//2):
#     b = []
#     zeros = [["" for i in range(delta[i][0].size)],["" for i in range(delta[i][1].size)]]
#     ma = max(square_r_err[i][::2].size,square_r_err[i+countFiles//2].size,square_r_err[i][::2].size)
#     b.append(["" if i%2==1 else 1+(i//2) for i in range((ma*2)-1)])
#     b.append(np.around(list(mi.roundrobin(square_err[i][0],Delta_err[i][0])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][0],Delta_err[i][0])))))
#     b.append(list(mi.roundrobin(np.around(delta_err[i][0],3),zeros[0]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta_err[i][0],zeros[0])))))
#     b.append(np.around(list(mi.roundrobin(square_err[i][1],Delta_err[i][1])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][1],Delta_err[i][1])))))
#     b.append(list(mi.roundrobin(np.around(delta_err[i][1],3),zeros[1]))+[""]*((ma*2)-1-len(list(mi.roundrobin(delta_err[i][1],zeros[1])))))
#     b.append(np.around(list(mi.roundrobin(square_err[i][2],Delta_err[i][2])),3).tolist()+[""]*((ma*2)-1-len(list(mi.roundrobin(square_err[i][2],Delta_err[i][2])))))
#     labels = ["Ordnung","innerste","","mittlere","","äußerste"]
#     b = np.array(b).T
#     fig.append(plt.figure())
#     ax.append(fig[i+countFiles//2].add_axes([0,0,1,1]))
#     ax[i+countFiles//2].table(cellText=b,colLabels=labels,loc='center',rowLoc='center')
#     ax[i+countFiles//2].axis("off")
#     ax[i+countFiles//2]
# 
# """ Plot speichern """
# for i in range(countFiles//2):
#     if i<12:
#         messung = 12 + i//3
#     else:
#         messung = 8 + i//3
#     farbe = farbArray[i%3]
#     fig[i].savefig(f"./1.Plots/{messung}{farbe}_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert
#     
# for i in range(countFiles//2):
#     if i<12:
#         messung = 12 + i//3
#     else:
#         messung = 8 + i//3
#     farbe = farbArray[i%3]
#     fig[i+countFiles//2].savefig(f"./1.Plots/{messung}{farbe}_error_table.png", dpi=100) # Bild als png Datei in Ordner Plots gespeichert
# =============================================================================

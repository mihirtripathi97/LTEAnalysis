import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from calc_tb import LTEAnalysis


line  = 'c18o'
Tex   = 30. # K
Ncol  = 1e16 #g cm^-2
Xconv = 1e-7
delv  = 0.5 # km/s

Ncols = np.logspace(14, 19, 128) # g cm^-2
#Texes = np.array([20, 30, 40, 50, 60]) # K
Jus   = [2, 3]

model = LTEAnalysis()
model.read_lamda_moldata(line)
#print (model.get_intensity(line, Ju, Tex, Ncols, delv, Xconv=Xconv, return_tau=False))


fig = plt.figure()
ax  = fig.add_subplot(111)

tau_1 = []
tau_2 = []
ratio = []
for i, Ncol_i in enumerate(Ncols):
    tau_1.append(model.get_intensity(line, Jus[0], Tex, Ncol_i, delv, Xconv=Xconv,
        return_tau=False, Tb=True))
    tau_2.append(model.get_intensity(line, Jus[1], Tex, Ncol_i, delv, Xconv=Xconv,
        return_tau=False, Tb=True))

    ratio.append(model.get_tbratio(line, Jus[1], Jus[0], Tex, Ncol_i, delv, Xconv=Xconv))

ax.plot(Ncols, tau_1, color='b', label=r'$T_\mathrm{b}$(%i-%i)'%(Jus[0], Jus[0]-1))
ax.plot(Ncols, tau_2, color='r', label=r'$T_\mathrm{b}$(%i-%i)'%(Jus[1], Jus[1]-1))
ax.plot(Ncols, ratio, color='k', 
    label=r'$T_\mathrm{b}$(%i-%i)/$T_\mathrm{b}$(%i-%i)'%(Jus[1], Jus[1]-1, Jus[0], Jus[0]-1))
ax.set_xscale('log')
ax.set_yscale('log')

ax.hlines(3, Ncols[0], Ncols[-1], linestyle='--', color='k') # 3 K
ax.set_xlabel(r'Column density (g cm$^{-2}$)')
#c=cm.gist_earth(float(j+1)/len(Ncols)))
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from calc_tb import LTEAnalysis
from pyfigures import change_aspect_ratio

#plt.rcParams['font.size'] = 24

# ----- input -----
line  = 'c18o'
Xconv = 1e-7
delv  = 0.5 # km/s
ilines = [6,3,3,2] # Ju
Ncols = np.logspace(16, 19, 128) # g cm^-2
Texes = np.array([10, 15., 20, 25, 30, 40, 50, 60]) # K
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
#print (model.get_intensity(line, Ju, Tex, Ncols, delv, Xconv=Xconv, return_tau=False))


fig = plt.figure()#figsize=(11.69, 8.27))
ax  = fig.add_subplot(111)

for i, Tex_i in enumerate(Texes):
    ratio_1 = []
    ratio_2 = []
    for j, Ncol_i in enumerate(Ncols):
        #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
        ratio_1.append(model.get_tbratio(line, ilines[0], ilines[1], Tex_i, Ncol_i, delv, 
            Xconv=Xconv))
        ratio_2.append(model.get_tbratio(line, ilines[2], ilines[3], Tex_i, Ncol_i, delv, 
            Xconv=Xconv))

    #print(ratio_32_21)
    ax.plot(ratio_1, ratio_2, c=cm.coolwarm(float(i+1)/len(Texes)))


for j, Ncol_i in enumerate([1e16, 1e17, 1e19]):
    ratio_1 = []
    ratio_2 = []
    for i, Tex_i in enumerate(np.linspace(Texes[0],Texes[-1],64)):
        #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
        ratio_1.append(model.get_tbratio(line, ilines[0], ilines[1], Tex_i, Ncol_i, delv, 
            Xconv=Xconv))
        ratio_2.append(model.get_tbratio(line, ilines[2], ilines[3], Tex_i, Ncol_i, delv, 
            Xconv=Xconv))

    #print(ratio_32_21)
    ax.plot(ratio_1, ratio_2, color='k')#c=cm.gist_earth(float(j+1)/len(Ncols)))

# case of tau=1

ax.set_xlabel(r'$T_\mathrm{b}$(%i-%i)/$T_\mathrm{b}$(%i-%i)'%(ilines[0], ilines[0]-1, ilines[1], ilines[1]-1))
ax.set_ylabel(r'$T_\mathrm{b}$(%i-%i)/$T_\mathrm{b}$(%i-%i)'%(ilines[2], ilines[2]-1, ilines[3], ilines[3]-1))

ax.set_xlim(0.3, 0.9)
ax.set_ylim(0.7, 1.5)

change_aspect_ratio(ax, 1)

fig.savefig('lineratio_lte_653221.pdf', transparent=True)
plt.show()

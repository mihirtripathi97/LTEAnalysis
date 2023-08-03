import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from calc_tb import LTEAnalysis
from pyfigures import change_aspect_ratio

#plt.rcParams['font.size'] = 24

# ----- input -----
line  = 'oh2co-h2'
Xconv = 1.e-9
delv  = 1. # km/s
#ilines = [7, 5, 4, 2] # Ju
Ncols = np.array([1e16, 1e19,]) # g cm^-2
Texes = np.array([10., 20, 30, 40, 50, 60]) # K
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
#print (model.get_intensity(line, Ju, Tex, Ncols, delv, Xconv=Xconv, return_tau=False))


for ilines in ([7, 5], [7, 4], [5, 4], [7, 2], [4, 2]):
    # figure
    fig = plt.figure()#figsize=(11.69, 8.27))
    ax  = fig.add_subplot(111)

    # iso-temperature curves
    for i, Tex_i in enumerate(Texes):
        ratio_1 = []
        ratio_2 = []
        for j, Ncol_i in enumerate(np.logspace(np.log10(Ncols[0]), np.log10(Ncols[-1]),128)):
            #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
            ratio_1.append(model.get_intensity(line, ilines[0], Tex_i, Ncol_i, delv, 
                Xconv=Xconv))
            ratio_2.append(model.get_intensity(line, ilines[1], Tex_i, Ncol_i, delv, 
                Xconv=Xconv))

        #print(ratio_32_21)
        ax.plot(ratio_1, ratio_2, c=cm.coolwarm(float(i+1)/len(Texes)))


    # iso-column curve
    for j, Ncol_i in enumerate(Ncols):
        ratio_1 = []
        ratio_2 = []
        for i, Tex_i in enumerate(np.linspace(Texes[0],Texes[-1],64)):
            #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
            ratio_1.append(model.get_intensity(line, ilines[0], Tex_i, Ncol_i, delv, 
                Xconv=Xconv))
            ratio_2.append(model.get_intensity(line, ilines[1], Tex_i, Ncol_i, delv, 
                Xconv=Xconv))

        #print(ratio_32_21)
        ax.plot(ratio_1, ratio_2, color='k')#c=cm.gist_earth(float(j+1)/len(Ncols)))

# case of tau=1

    trns = [ str(
        model.moldata[line]['J'][
        int(model.moldata[line]['Jup'][ilines[i] - 1] - 1)]
        )
        + '-'
        + str(
            model.moldata[line]['J'][
            int(model.moldata[line]['Jlow'][ilines[i] - 1] - 1)]
        )
        for i in range(len(ilines))
    ]
    #print(model.moldata[line]['Jup'][ilines[0]])
    trns_label = [
        '$%s_{%s,%s}$–$%s_{%s,%s}$'%(list(trns[i])[0], 
        list(trns[i])[1], list(trns[i])[2], list(trns[i])[4], 
        list(trns[i])[5], list(trns[i])[6])
        for i in range(len(trns))]
    ax.set_xlabel(r'$T_\mathrm{b}$(%s)'%(trns_label[0]))
    ax.set_ylabel(r'$T_\mathrm{b}$(%s)'%(trns_label[1]))

    ax.set_xlim(3., 20.)
    ax.set_ylim(3., 20.)

    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([5, 10, 15, 20])
    change_aspect_ratio(ax, 1)

    fig.subplots_adjust(bottom =0.15)
    fig.savefig('lte-lineratio_h2co_%s.pdf'%(trns[0] + '_vs_' + trns[1]), transparent=True)
    #plt.show()

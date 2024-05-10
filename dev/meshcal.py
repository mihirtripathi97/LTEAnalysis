import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
sys.path.append('..')
plt.rcParams['font.size'] = 12

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.5e5 # cm/s
ilines = [3, 2] # Ju
Ncols = np.array([1.e21, 1.e22, 1.e23, 1.e24,]) * Xconv # cm^-2
Texes = np.array([5., 10., 20., 30., 40., 60., 300.]) # K
return_tau = False
# ------------------


# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
for Ncol, Tex in itertools.product(Ncols, Texes):
    Tb32 = model.get_intensity(line, 3, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau = return_tau)
    Tb21 = model.get_intensity(line, 2, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau = return_tau)
    tau32 = model.get_intensity(line, 3, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau = True)
    tau21 = model.get_intensity(line, 2, Tex, Ncol, delv, lineprof='gauss', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau = True)

    print('(Ncol, Tex)=(%.2e, %.1f)'%(Ncol, Tex))
    print('   (tau21, Tb 2-1)=(%.4e, %.4f)'%(tau21, Tb21))
    print('   (tau32, Tb 3-2)=(%.4e, %.4f)'%(tau32, Tb32))
    #if return_tau:
    #    print('(Ncol, Tex)=(%.2e, %.1f) : (tau 3-2, tau 2-1)=(%.2e, %.2e)'%(Ncol, Tex, Tb32, Tb21))
    #else:
    #    print('(Ncol, Tex)=(%.2e, %.1f) : (Tb 3-2, Tb 2-1)=(%.2f, %.2f)'%(Ncol, Tex, Tb32, Tb21))


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'ph2co-h2'
Xconv = 1e-10
delv  = 0.5e5 # cm/s
ilines = [[10,3], [13, 3]] # Ju
Ncols = np.array([1.e22, 1.e24, 1e26]) # cm^-2
Texes = np.array([10., 20., 30., 40., 60, 80, 100, 120,]) # K
# Example of data
data = [6., 6.,] # Tb3-2, Tb2-1
e_data = [1., 0.5] # Error of data
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
fig, ax = model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, 
    Xconv=Xconv, lw=1., xtype = 'ratio', ytype = 'ratio')
ax.set_xlim(0., 1.2)
ax.set_ylim(0., 1.2)

ax.errorbar(data[0], data[1], xerr=e_data[0], yerr=e_data[1],
    color='k', capsize=2., capthick=2., marker='o')

plt.show()
# -----------------------
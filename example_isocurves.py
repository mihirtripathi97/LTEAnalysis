import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.5 # km/s
ilines = [3,2] # Ju
Ncols = np.array([1.e16, 1.e17, 1.e18, 1.e19,]) # cm^-2
Texes = np.array([5., 10., 15., 20., 30., 40.]) # K
# Example of data
data = [6., 6.,] # Tb3-2, Tb2-1
e_data = [1., 0.5] # Error of data
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
fig, ax = model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, Xconv=Xconv, lw=1.)
ax.set_xlim(0., 15)
ax.set_ylim(0., 15)

ax.errorbar(data[0], data[1], xerr=e_data[0], yerr=e_data[1],
    color='k', capsize=2., capthick=2., marker='o')

plt.show()
# -----------------------
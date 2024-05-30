import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.07e5 # cm/s
ilines = [3,2] # Ju
Ncols = np.array([5.e13, 5e14, 5.e15, 9.e15, 5e16, 1.e17]) # cm^-2
Texes = np.array([5., 10., 15., 20., 25., 30., 40.]) # K
# Example of data
data = [6., 6.,] # Tb3-2, Tb2-1
e_data = [1., 0.5] # Error of data
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
fig, ax = model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, lw=1.) # Xconv=Xconv,
ax.set_xlim(0., 20)
ax.set_ylim(0., 20)

ax.errorbar(data[0], data[1], xerr=e_data[0], yerr=e_data[1],
    color='k', capsize=2., capthick=2., marker='o')

plt.grid()

plt.show()
# -----------------------
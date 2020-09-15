import sys
import matplotlib.pyplot as plt

import numpy as np
import itertools
params = {'legend.fontsize': 13,
          'legend.handlelength': 1}
plt.rcParams.update(params)
xa = [4, 5, 6, 7, 8]
yae = [85.58, 76.41, 76.80, 79.91, 79.94]
y_errorae = [1.48, 0.37, 0.33, 0.66, 1.45]
markers=itertools.cycle(('d', '*')) 
#slabel = itertools.cycle(('Random', 'Passive', 'ITRS'))
#plt.style.use('seaborn-whitegrid')
xb = [20, 25, 30, 35, 40]
ybe = [77.5,76.86,76.80,78.07,77.49]
y_errorbe = [0.76,0.86,0.33, 0.6, 1.54]

yam = [80.27, 81.25, 83.78, 83.10, 84.6]
y_erroram = [1.04, 2.02, 1.23, 1.44, 1.78]
ybm = [81.62,82.51,83.78,81.58,80.86]
y_errorbm = [1.02,1.41,1.23, 2.01, 2.78]

yal = [81.11, 83.63, 85.55, 84.51, 85.47]
y_erroral = [1.93, 2.77, 1.68, 1.15, 2.01]
ybl = [84.78,83.84,85.55,84.54,82.05]
y_errorbl = [0.69,1.19,1.68, 1.93, 2.24]

fig, ax = plt.subplots(2,3)
#plt.xlim(0, num_batch-1)
#plt.ylim(0.55, 0.9)
#ax[0][0].set_xlabel("Alpha", fontsize = 10)
#ax[0][1].set_xlabel("Alpha", fontsize = 10)
ax[0][1].set_xlabel("Alpha", fontsize = 15)
#ax[1][0].set_xlabel("Beta", fontsize = 10)
#ax[1][1].set_xlabel("Beta", fontsize = 10)
ax[1][1].set_xlabel("Beta", fontsize = 15)
#ax[0][2].xaxis.set_label_coords(1, -0.05)
#ax[1][2].xaxis.set_label_coords(1.15, -0.05)
ax[0][0].errorbar(xa, yae, yerr = y_errorae, marker = 'd', label = 'Beta = 30', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'green', uplims=True, lolims=True)
    
ax[0][1].errorbar(xa, yam, yerr = y_erroram, marker = 'd', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'green', uplims=True, lolims=True)

ax[0][2].errorbar(xa, yal, yerr = y_erroral, marker = 'd', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'green', uplims=True, lolims=True)

ax[1][0].errorbar(xb, ybe, yerr = y_errorbe, marker = '*', label = 'Alpha = 6', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'purple', uplims=True, lolims=True)
    
ax[1][1].errorbar(xb, ybm, yerr = y_errorbm, marker = '*', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'purple', uplims=True, lolims=True)

ax[1][2].errorbar(xb, ybl, yerr = y_errorbl, marker = '*', mec = 'black', markersize = 10, fmt = '-', elinewidth = 2, color = 'purple', uplims=True, lolims=True)



ax[0][0].set_ylim(75,88)
ax[0][1].set_ylim(75,88)
ax[0][2].set_ylim(75,88)
ax[1][0].set_ylim(75,88)
ax[1][1].set_ylim(75,88)
ax[1][2].set_ylim(75,88)

ax[0][1].get_yaxis().set_visible(False)
ax[0][2].get_yaxis().set_visible(False)
ax[1][1].get_yaxis().set_visible(False)
ax[1][2].get_yaxis().set_visible(False)

#ax[0][0].get_xaxis().set_visible(False)
#ax[0][1].get_xaxis().set_visible(False)
#ax[0][2].get_xaxis().set_visible(False)

ax.flat[0].set_title("Early phase", fontsize = 19)
ax.flat[1].set_title("Middle phase", fontsize = 19)
ax.flat[2].set_title("Late phase", fontsize = 19)

    
fig.text(0.04, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize = 15)
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# finally we invoke the legend (that you probably would like to customize...)
#ax[0][0].xticks(fontsize=13)
#fig.yticks(fontsize=13)
fig.legend(lines, labels, ncol = 2, loc = 'upper center')
fig.savefig('param.pdf')

plt.show()


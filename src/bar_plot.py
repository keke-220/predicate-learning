import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def sort_by_index(a, seq):
    new_list = []
    for i in seq:
       new_list.append(a[i])
    return new_list

seq = [5, 6, 3, 9, 0, 8, 1, 7, 2, 4]
#men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
#women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)
batch1 = [734.67,755,702.67,637.33,648,656.67,637.33,720.33,744.33,712]
batch1_std = [25.17,14,55.08,44.38,30.05,43.66,49.69,21.36,27.5,56.4]
batch5 = [692.67,755,959,629.33,941.67,631.67,672,895.33,801.33,811]
batch5_std = [48.79,62.75,19,28.29,53.69,31.75,81.66, 17.01,50.12,88.1]
batch10 = [845.67,864,952.67,783.67,965.67,614,710,902.33,849,795]
batch10_std = [25.48,21.7,58.77,84.4,3.06,71.25,89.21,10.07,25.98,31.43]
batch1 = sort_by_index(batch1, seq)
batch1_std = sort_by_index(batch1_std, seq)
batch5 = sort_by_index(batch5, seq)
batch5_std = sort_by_index(batch5_std, seq)
batch10 = sort_by_index(batch10, seq)
batch10_std = sort_by_index(batch10_std, seq)

for i in range(len(batch1)):
    batch1[i] = batch1[i]/10
for i in range(len(batch1_std)):
    batch1_std[i] = batch1_std[i]/10

for i in range(len(batch5)):
    batch5[i] = batch5[i]/10

for i in range(len(batch5_std)):
    batch5_std[i] = batch5_std[i]/10

for i in range(len(batch10)):
    batch10[i] = batch10[i]/10

for i in range(len(batch10_std)):
    batch10_std[i] = batch10_std[i]/10


labels = ['soft', 'green', 'full', 'empty', 'container', 'plastic', 'hard', 'blue', 'metal', 'toy']
labels = sort_by_index(labels, seq)




degrees = 45



ind = np.arange(len(batch1))  # the x locations for the groups
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, batch1, width, yerr=batch1_std, label='Random', hatch = "//", color = '#AFBADC')
rects2 = ax.bar(ind, batch5, width, yerr=batch5_std, label='Passive', hatch = "\\\\", color = '#F4E3B1')
rects3 =  ax.bar(ind + width, batch10, width, yerr=batch10_std, label='ITRS', hatch = "--", color = '#8ED3F4')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)', fontsize= 14)
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(labels, fontsize = 14, rotation = degrees)
ax.legend()

fig.tight_layout()

plt.show()

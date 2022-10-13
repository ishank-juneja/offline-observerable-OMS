import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['hatch.linewidth'] = 3
plt.rcParams['hatch.color'] = 'gray'

# Avg here is average with R=5 and 
# L1 here is L1 with CR = 0.8
x_markers = ["Avg", "DL", "L1", "SS"]

df = pd.read_csv("resource_csv.csv", index_col=0, delimiter=',', skipinitialspace=True)

# fig = plt.figure(figsize=(18, 8)) # Create matplotlib figure
fig = plt.figure() # Create matplotlib figure

# ax is the main axis and ax2 is the dependent axis
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

df.Time.plot(kind='bar', color='green', ax=ax, width=width, position=1.25, yerr=list(df.TimeStddev), error_kw=dict(ecolor='black', lw=1.5, capsize=5, capthick=1.5))
df.Memory.plot(kind='bar', color='orange', ax=ax2, width=width, position=0.25, yerr=list(df.MemoryStddev), error_kw=dict(ecolor='black', lw=1.5, capsize=5, capthick=1.5), hatch="//")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# finally we invoke the legend (that you probably would like to customize...)

plt.legend(lines, labels, loc="upper left", fontsize=12)

ax.set_ylabel('Time (s)', fontsize=18, rotation=90, labelpad=-5)
ax2.set_ylabel('Memory (MB)', fontsize=18, rotation=270, labelpad=25)


plt.xlim([-0.75, 3.5])

plt.xticks(np.arange(-0.1, 3.9, 1), x_markers)
ax.set_xticklabels(x_markers, rotation=0, fontsize=18)
ax.set_yticks(np.arange(0, 121, 20))
ax.set_yticklabels(np.arange(0, 121, 20), fontsize=14)
ax2.set_yticks(np.arange(0, 3501, 500))
ax2.set_yticklabels(np.arange(0, 3501, 500), fontsize=14)

plt.grid()

#ax.margins(x=None, y=1)

#ax.set_xticklabels(fontsize=14)
#ax2.set_xticklabels(fontsize=14)

#plt.legend(handles=[bar1])

#plt.grid(axis = 'y')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.savefig("resource.eps", bbox_inches="tight")

plt.close()

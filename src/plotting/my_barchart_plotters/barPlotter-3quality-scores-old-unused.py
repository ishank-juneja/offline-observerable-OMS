import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x_markers = ["Unstable", "Averaging", "Deep Online", "L1 method", "SubSpace"]
yvals_time = [2.052, 8.325, 4.679, 64.204]
yvals_memory = [98.1, 2210.6, 128.8, 1715.1]

df = pd.read_csv("quality_csv.csv", index_col=0, delimiter=',', skipinitialspace=True)

fig = plt.figure(figsize=(14, 4.8)) # Create matplotlib figure

# ax is the main axis and ax2 is the dependent axis
ax = fig.add_subplot(121) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

# ax is the main axis and ax2 is the dependent axis
ax3 = fig.add_subplot(122) # Create matplotlib axes
ax4 = ax3.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

df.SSIM.plot(kind='bar', color='khaki', ax=ax, width=width, position=0)
df.PSNR.plot(kind='bar', color='gray', ax=ax2, width=width, position=1)
df.Rhythm.plot(kind='bar', color='orange', ax=ax3, width=width, position=1)
df.Distance.plot(kind='bar', color='green', ax=ax4, width=width, position=0)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# finally we invoke the legend (that you probably would like to customize...)

ax.legend(lines, labels, loc=(0.45, 0.84), fontsize=12)

ax.yaxis.grid()


ax3.set_ylabel('Rhythm', fontsize=18)
ax4.set_ylabel('Distance', fontsize=18)
ax.set_ylabel('SSIM', fontsize=18)
ax2.set_ylabel('PSNR', fontsize=18)

plt.xticks(np.arange(-0.1, 4.9, 1), x_markers)
ax.set_xticklabels(x_markers, rotation=0, fontsize=12)
ax4.set_ylim([0, 6000])

ax.set_xlim([-0.6, 4.6])

ax.set_yticks(np.round(np.arange(0, 0.40, 0.050), 2))
ax.set_yticklabels(np.round(np.arange(0, 0.40, 0.050), 2), fontsize=14)
ax2.set_yticks(np.arange(0, 15, 2))
ax2.set_yticklabels(np.arange(0, 15, 2), fontsize=14)

ax3.set_yticks(np.arange(0, 0.071, 0.01))
ax3.set_yticklabels(np.arange(0, 0.071, 0.01), fontsize=14)
ax4.set_yticks(np.arange(0, 6001, 1000))
ax4.set_yticklabels(np.arange(0, 6001, 1000), fontsize=14)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[2:]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# finally we invoke the legend (that you probably would like to customize...)

ax3.legend(lines, labels, loc=(0.54, 0.84), fontsize=12)

plt.xticks(np.arange(-0, 5.0, 1), x_markers)
ax.set_xticklabels(x_markers, rotation=0, fontsize=12)
ax3.set_xticklabels(x_markers, rotation=0, fontsize=12)

ax3.set_xlim([-0.6, 4.6])

ax3.yaxis.grid()

#ax.set_xticklabels(fontsize=14)
#ax2.set_xticklabels(fontsize=14)

#plt.legend(handles=[bar1])

#plt.grid(axis = 'y')

fig.tight_layout(pad=1.5)

plt.savefig("quality.eps", bbox_inches="tight")

plt.close()

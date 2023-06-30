import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='ticks', font_scale=1.1)

def to_adsorption(res_ib) :
	return np.sum(res_ib, axis=1)

def get_avg_length(ads) :
	ads_sum_t = np.sum(ads, axis=1)
	p_ads = np.array([ads[t] / ads_sum_t[t] for t in range(tsteps)])
	p_ads[0,0] = 1; p_ads[0,1:] = np.zeros(max_size-1)
	avg_length = np.sum([np.arange(1, max_size+1) * p_ads[t] for t in range(tsteps)], axis=1)
	# avg_length *= ... 	# Convert from oligomer units units to nm?

	return avg_length

omega = 6.1e5

rads = [1, 5]
nbs = [6.25, 12.5, 25, 50]

tsteps = 100
max_size = 60

res = np.empty((len(rads), len(nbs), tsteps, max_size))
for i_r, rad in enumerate(rads) :
	for i_b, nb in enumerate(nbs) :
		res[i_r, i_b, ...] = np.loadtxt(f'res_r{rad}_b{nb}_sp.txt')

sizes = np.arange(1, max_size+1)

fig, axs = plt.subplots(2, 3, figsize=(12,8), constrained_layout=True)

colors = ['magenta', 'green', 'blue', 'orange']
tspan = (np.linspace(0, 1, tsteps) * 24000 / 60).astype(int)
tspan_bi = (np.linspace(0, 1, tsteps) * 96000 / 60).astype(int)

# Plot adsorption over time
for i_r, rad in enumerate(rads) :
	ax_r = axs[0,i_r]
	for i_b, nb in enumerate(nbs) :
		ax_r.plot(tspan, np.sum(sizes * res[i_r,i_b], axis=1), color=colors[i_b], label=f"{nb} nM")
	ax_r.set_title(f"Mono-dispersed {rad}$\\mu$m beads")
	ax_r.set_xlabel("Time (min)")
	ax_r.set_ylabel("Adsorption (a.u.)")

# Plot average length over time
for i_r, rad in enumerate(rads) :
	ax_r = axs[1,i_r]
	for i_b, nb in enumerate(nbs) :
		avg_length = get_avg_length(res[i_r,i_b])

		ax_r.plot(tspan, avg_length, color=colors[i_b], label=f"{nb} nM")
	ax_r.set_xlabel("Time (min)")
	ax_r.set_ylabel("Average length (oligomer units)")

# Plot results for bidispersed
res_bi = np.loadtxt('res_bi.txt')
res = np.empty((2, tsteps, max_size))
res[0] = res_bi[:,:60]
res[1] = res_bi[:,60:]

axs[0,2].plot(tspan, np.sum(sizes * res[0], axis=1), color='magenta', label='$1\\mu m$')
axs[0,2].plot(tspan, np.sum(sizes * res[1], axis=1), color='blue', label='$5\\mu m$')
axs[0,2].set_title(f"Bi-dispersed of $n_b^0=25$ nM")
axs[0,2].set_xlabel("Time (min)")
axs[0,2].set_ylabel("Adsorption (a.u.)")

axs[1,2].plot(tspan, get_avg_length(res[0]), color='magenta', label='$1\\mu m$')
axs[1,2].plot(tspan, get_avg_length(res[1]), color='blue', label='$5\\mu m$')
axs[1,2].set_xlabel("Time (min)")
axs[1,2].set_ylabel("Average length (oligomer units)")

for ax in axs.flatten() :
	# breakpoint()
	ax.axvline(90, ls='--', color='gray', alpha=0.5, zorder=0)
	ax.legend(frameon=False, reverse=True)

plt.savefig('fig_main.pdf', bbox_inches='tight')

plt.show()
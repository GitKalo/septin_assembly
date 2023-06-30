import math, time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### Define helper functions
def get_anneal(n, n_rev, i) :	# Test speed
	idx = i-1
	return np.sum(n[:idx] * n_rev[-idx:])

### Static parameters
# k_bind_0 = {1: 4.41, 5: 0.86}
k_bind_0 = {1: 4.41*1.66, 5: 0.86*1.66}		# As def in author code
k_unbind = {1: 9.72, 5: 11.54}
# k_unbind = {1: 10, 5: 10}		# As def in author code
h_bead_inv = 1/2e4
# l_avg = {1: 328, 5: 467}	# in nm

# beta = {1: 0.044, 5: 0.1}
beta = (0.044 + 0.1)/2 	# Take as average
k_bind_coop = 3.1e-3
k_anneal = 2e-7
k_frag = {1: 5.1e-4, 5: 2.5e-4}
xi = {1: 3, 5: 2}
omega = 6.1e5
omega_inv = 1/omega
# n_sat = {1: 1.2e5, 5: 2.9e5}
# n_sat = {1: 0.2076*1.05, 5: 0.2076*1.05}	# As def in author code
# n_sat = {1: 1.2e5/omega, 5: 2.9e5/omega}	# As def in author code (almost)
n_sat = (1.2e5/omega + 2.9e5/omega)/2 	# Take as average

### Functional parameters
# Run and assess practical efficiency; if needed, implement via derivatives
n_s = lambda n : np.sum(sizes * n, axis=1)	# Number of bound oligomers per radius
n_bulk = lambda n, ns : (n_bulk_0 - h_bead_inv*ns*omega) * (n_bulk_0 > h_bead_inv*ns*omega)
n_bulk_eff = lambda nb, ns : (nb / (1+beta*nb)) * (1-ns/n_sat*(ns < n_sat)-(ns > n_sat))

J_bind_direct = lambda nbeff, rad : k_bind_0[rad] * nbeff * omega_inv
J_bind_coop_1 = lambda n, nbeff : k_bind_coop*nbeff * (np.sum(n[2:]*(sizes[2:]-2)) - 2*n[0])
J_bind_coop_i = lambda n, nbeff : 2*k_bind_coop*nbeff * (n[:-1]-n[1:])
J_anneal_1 = lambda n : -n[0]*k_anneal*np.sum(n[:-1]) * omega
J_anneal_i = lambda n, n_rev : k_anneal*(0.5*np.array([get_anneal(n, n_rev, i) for i in sizes[1:]]) - n[1:]*np.array([np.sum(n[:-i]) for i in sizes[1:]])) * omega
J_frag_1 = lambda n, nsum, rad : k_frag[rad] * (nsum-n[0])
J_frag_i = lambda n, rad : -0.5*n[1:]*k_frag[rad]*(sizes[1:]-1) + k_frag[rad]*(np.array([np.sum(n[i+1:]) for i in sizes[1:]]))
J_unbind_1 = lambda n, rad : -k_unbind[rad]*n[0]
J_unbind_i = lambda n, rad : -k_unbind[rad]*np.exp(-xi[rad]*(sizes[1:]-1))*n[1:]

### Differential equations and integration function
# Authors use max=40 (radius 1) and max=60 (radius 5)
max_size = 60
sizes = np.arange(1, max_size + 1) 	# Size index (n index + 1)
n0 = np.zeros((2, max_size))	# Initialize with no bound septins

def dn(n) :		# Try implementing with Numba??
	'''
	Main kinetic ODEs.

	In this case, n is a flattened array with 2*max_size elements, representing
	adsorption on beads of the two different radii.
	
	Idea is to use the different binding rates specific to bead curvature, 
	but the same (depleting) bulk density for beads of all curvatures.
	'''
	n = n.reshape((2, max_size))
	d_n = np.zeros_like(n) 	# Default rate zero

	nsum = np.sum(n, axis=1)
	ns = n_s(n)			# Shape (2,...)
	nb = n_bulk(n, np.sum(ns))	
	nbeff = n_bulk_eff(nb, np.sum(ns))

	n_rev = n[:,::-1]

	### Length 1 filaments
	# Radius 1
	d_n[0, 0] = J_bind_direct(nbeff, 1) + J_bind_coop_1(n[0], nbeff) + J_anneal_1(n[0]) + J_frag_1(n[0], nsum[0], 1) + J_unbind_1(n[0], 1)
	# Radius 5
	d_n[1, 0] = J_bind_direct(nbeff, 5) + J_bind_coop_1(n[1], nbeff) + J_anneal_1(n[1]) + J_frag_1(n[1], nsum[1], 5) + J_unbind_1(n[1], 5)

	### Length 2 to max filaments
	# Radius 1
	d_n[0, 1:] = J_bind_coop_i(n[0], nbeff) + J_anneal_i(n[0], n_rev[0]) + J_frag_i(n[0], 1) + J_unbind_i(n[0], 1)
	# Radius 5
	d_n[1, 1:] = J_bind_coop_i(n[1], nbeff) + J_anneal_i(n[1], n_rev[1]) + J_frag_i(n[1], 5) + J_unbind_i(n[1], 5)

	d_n = d_n.reshape((2*max_size,))

	return d_n

if __name__ == '__main__' :

	rads = [1, 5] 		# Bead radius
	n_bulk_0 = 25		# Bulk densities (at start)

	n_bulk_baseline = n_bulk_0
	n_bulk_0 = n_bulk_0 * 0.6022

	tic = time.time()

	# Authors do time in np.linspace(0,5400,100) => 5400 seconds = 90 mins (at intervals of 5400/100 = 54s)
	time_lim = 96000
	# res_n = run_heun(n_write=5400, h=0.1)
	res = sp.integrate.solve_ivp(lambda t, y : dn(y), (0, time_lim), n0.reshape((2*max_size)), t_eval=np.linspace(0, time_lim, 100), method='BDF', rtol=1e-10, atol=1e-10)

	assert res.success
	res_n = res.y.T

	toc = time.time()
	print(f"Finished run for bidispersed and nb0={n_bulk_baseline} in {toc-tic:.2f}s")

	np.savetxt(f'res_bi.txt', res_n)

breakpoint()
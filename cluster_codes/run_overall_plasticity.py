#!/usr/bin/python

#PBS -j oe
#PBS -J 1-600:1
#PBS -o /storage/ofiles/rdp
#PBS -e /storage/efiles

import os
import numpy as np

clus_index = int(os.getenv('PBS_ARRAY_INDEX')) - 1
#print clus_index


n_RyR = np.arange(10, 61, 10)#param1 = []
tau_refill = np.array((1e-6, 1e-3, 1e-1, 1e0, 1e1))#param2 = []
f_input = np.arange(0.1, 20.1, 0.1)
g_NMDAR = np.array((40.0, 80.0, 120.0, 160.0, 200.0))
t = 10 

params = []
for r in n_RyR:
	for f in f_input:
		for g in g_NMDAR:
			params.append(str(r)+'_'+str(t)+'_'+str(f)+'_'+str(g))

os.system('python /home/vikrant/overall_plasticity.py '+params[clus_index])

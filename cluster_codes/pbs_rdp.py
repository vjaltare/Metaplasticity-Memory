#!/usr/bin/python

#PBS -j oe
#PBS -J 1-600:1
#PBS -o /storage/ofiles/rdp
#PBS -e /storage/efiles

import os
import numpy as np

clus_index = int(os.getenv('PBS_ARRAY_INDEX')) - 1
print clus_index


n_RyR = np.arange(10, 61, 10)#param1 = []
tau_refill = np.array((1e-6, 1e-3, 1e-1, 1e0, 1e1))#param2 = []
f_input = np.arange(0.1, 20.1, 0.1)

params = []
for r in n_RyR:
	for t in tau_refill:
		for f in f_input:
			params.append(str(r)+'_'+str(t)+'_'+str(f))

os.system('python /home/vikrant/ryr_analysis_rdp.py '+params[clus_index])

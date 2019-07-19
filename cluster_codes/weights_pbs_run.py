#!/usr/bin/python

#PBS -j oe
#PBS -J 1-1200:1
#PBS -o /storage/vikrant/rdp_plasticity
#PBS -e /storage/vikrant/efiles

import os
import numpy as np

clus_index = int(os.getenv('PBS_ARRAY_INDEX')) - 1


n_RyR = np.arange(0, 70, 10)#param1 = []
#tau_refill = np.array((1e-6, 1e-3, 1e-1, 1e0, 1e1))#param2 = []
f_input = np.arange(0.1, 20.1, 0.1)
t = 1 #s
params = []

for r in n_RyR:
	for f in f_input:
		params.append(str(r)+'_'+str(t)+'_'+str(f))

os.system('python /home/vikrant/weights_rdp.py '+params[clus_index])

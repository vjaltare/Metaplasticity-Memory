#!/usr/bin/python

#PBS -j oe
#PBS -J 1-81:1
#PBS -o /storage/ofiles
#PBS -e /storage/efiles

import os

clus_index = int(os.getenv('PBS_ARRAY_INDEX')) - 1
print clus_index


l1 = range(1, 10)#param1 = []
l2 = range(10, 20)#param2 = []

params = []
for p1 in l1:
	for p2 in l2:
		params.append(str(p1)+' '+str(p2))

os.system('python /home/vikrant/sampleCode.py '+params[clus_index])
